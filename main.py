from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt

class PathPlanner:
    def __init__(self):
        #BSpline
        self.BSpline_deg = 2  #degree
        self.BSpline_num = 8  #control point num
        self.BSpline_P = NULL #control point (3d vector)
        self.knot = NULL      #kont vector
        self.BSpoint = NULL   #x,y,z trajectory
        self.BSvel = NULL     #x,y,z velocity
        self.N = NULL         #basis function value

        #polynominal
        self.Polpoint = NULL
    
    def setBSplineParam(self, deg, P):
        self.BSpline_deg = deg
        self.BSpline_num = len(P)
        self.BSpline_P = P
        self.knot = np.zeros(self.BSpline_num+self.BSpline_deg+1)
        for i in range(0, self.BSpline_deg+1):
            self.knot[i] = 0
        for i in range(self.BSpline_deg+1, self.BSpline_num):
            self.knot[i] = (i-self.BSpline_deg)/(self.BSpline_num-2)
        for i in range(self.BSpline_num, self.BSpline_deg+self.BSpline_num+1):
            self.knot[i] = 1
    
    def b(self, i, k, alpha):
        #i : [0, num-1]
        #k : [0, deg]
        #alpha : [0, 1]
        if k == 0:
            return 1.0 if self.knot[i] <= alpha <= self.knot[i+1] else 0.0
        else:
            Np1 = self.b(i, k-1, alpha)
            Np2 = self.b(i+1, k-1, alpha)
            if self.knot[i+k] == self.knot[i]: #define 0/0 = 0
                fterm = 0.0
            else:
                fterm = ((alpha - self.knot[i])/(self.knot[i+k] - self.knot[i])) * Np1
            if self.knot[i+k+1] == self.knot[i+1]: #define 0/0 = 0
                sterm = 0.0
            else:
                sterm = ((self.knot[i+k+1] - alpha) / (self.knot[i+k+1] - self.knot[i+1])) * Np2
            return fterm + sterm           
    
    def BSpline(self, alpha):
        #alpha is the mapped vector from t ∈[T0=0,Tf] to [0,1]
        #Tf is the task completion time
        self.N = np.zeros((self.BSpline_num, len(alpha)))
        self.BSpoint = np.zeros((3, len(alpha)))

        for j in range(len(alpha)):
            for i in range(self.BSpline_num):
                if j == len(alpha) - 1:
                    self.N[i, j] = self.b(i, self.BSpline_deg, self.knot[-2])
                else:
                    self.N[i, j] = self.b(i, self.BSpline_deg, alpha[j])
                self.BSpoint[0, j] += self.N[i, j]*self.BSpline_P[i][0]
                self.BSpoint[1, j] += self.N[i, j]*self.BSpline_P[i][1]
                self.BSpoint[2, j] += self.N[i, j]*self.BSpline_P[i][2]
    
    def fifthPolynominal(self, T):
        #Tf is the task completion tima
        self.Polpoint = np.zeros(len(T))

        a3 = 10/(T[-1]**3)
        a4 = -15/(T[-1]**4)
        a5 = 6/(T[-1]**5)

        for i in range(len(T)):
            self.Polpoint[i] = a3*T[i]**3 + a4*T[i]**4 + a5*T[i]**5        
        

if __name__=="__main__":
    planner = PathPlanner()
    #define control point
    P = np.array([[3,3,0], [6,1,0], [9,1,0], [9,5,0], [12,5,0], [12,8,0], [6,5,0], [12,0,0]])
    #path planner Bspline set parameters
    #degree must be larger than 2
    planner.setBSplineParam(3, P)
    #make time parameter alpha
    # Tf = 10
    # t = np.linspace(0, Tf, 10000)
    # planner.fifthPolynominal(t)
    # alpha = planner.Polpoint
    alpha = np.linspace(0, 1, 10000)
    #path planning
    planner.BSpline(alpha)

    #visualize
    plt.figure(1)
    plt.plot(alpha, planner.N.T)
    plt.xlabel('$\eta$')
    plt.ylabel('$N(\eta)$')
    plt.title('B-Spline Basis Function')
    
    plt.figure(2)
    plt.plot(P[:, 0], P[:, 1], 'k')
    plt.plot(P[:, 0], P[:, 1], 'ro')
    plt.plot(planner.BSpoint[0, :], planner.BSpoint[1, :], 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('B-Spline Curve')
    plt.legend(['Control polygon', 'B-Spline Curve', 'Control Point'])
    plt.show()

    #微分が正しいか計算
    dx = np.zeros(len(alpha))
    dx[0] = 0.0
    for i in range(1, len(alpha)):
        dx[i] = (planner.BSpoint[0, i] - planner.BSpoint[0, i-1])/(t[i]-t[i-1])
    plt.figure(3)
    plt.plot(alpha, dx, label='nomal')
    plt.legend()
    plt.show()