from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt

class BSplinePlanner:
    def __init__(self):
        #BSpline
        self.BSpline_deg = 2  #degree
        self.BSpline_num = 8  #control point num
        self.BSpline_P = NULL #control point (3d vector)
        self.knot = NULL      #kont vector
        self.BSpoint = NULL   #x,y,z trajectory
        self.BSvel = NULL     #x,y,z velocity
        self.BSacc = NULL     #x,y,z acceleration
        self.BSjerk = NULL    #x,y,z jerk(da/dt)
        self.N = NULL         #basis function value
    
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
    
    def db(self, t, i, k, alpha):
        if t==0:
            return self.b(i, k, alpha)
        else:
            dNp1 = self.db(t-1, i, k-1, alpha)
            dNp2 = self.db(t-1, i+1, k-1, alpha)
            if(self.knot[i+k] == self.knot[i]):#define k/0 = 0
                dfterm = 0.0
            else:
                dfterm = k/(self.knot[i+k] - self.knot[i])*dNp1
            if(self.knot[i+k+1] == self.knot[i+1]):
                dsterm = 0.0
            else:
                dsterm = k/(self.knot[i+k+1] - self.knot[i+1])*dNp2
            return dfterm - dsterm
    
    def dBSpline(self, alpha):
        #alpha is the mapped vector from t:[0, Tf] to [0,1]
        #Tf is the task completion time
        dN = np.zeros((self.BSpline_num, len(alpha)))
        self.BSvel = np.zeros((3, len(alpha)))

        for j in range(len(alpha)):
            for i in range(self.BSpline_num):#Sigma^{BSnum-1}_{i=0} P_i * b^t_{i,k}(alpha(j))
                if j==len(alpha)-1:
                    dN[i, j] = self.db(1, i, self.BSpline_deg, self.knot[-2])
                else:
                    dN[i, j] = self.db(1, i, self.BSpline_deg, alpha[j])
                self.BSvel[0, j] += dN[i, j]*self.BSpline_P[i][0]
                self.BSvel[1, j] += dN[i, j]*self.BSpline_P[i][1]
                self.BSvel[2, j] += dN[i, j]*self.BSpline_P[i][2]

    def ddBSpline(self, alpha):
        #alpha is the mapped vector from t:[0, Tf] to [0,1]
        #Tf is the task completion time
        ddN = np.zeros((self.BSpline_num, len(alpha)))
        self.BSacc = np.zeros((3, len(alpha)))

        for j in range(len(alpha)):
            for i in range(self.BSpline_num):#Sigma^{BSnum-1}_{i=0} P_i * b^t_{i,k}(alpha(j))
                if j==len(alpha)-1:
                    ddN[i, j] = self.db(2, i, self.BSpline_deg, self.knot[-2])
                else:
                    ddN[i, j] = self.db(2, i, self.BSpline_deg, alpha[j])
                self.BSacc[0, j] += ddN[i, j]*self.BSpline_P[i][0]
                self.BSacc[1, j] += ddN[i, j]*self.BSpline_P[i][1]
                self.BSacc[2, j] += ddN[i, j]*self.BSpline_P[i][2]

    def dddBSpline(self, alpha):
        #alpha is the mapped vector from t:[0, Tf] to [0,1]
        #Tf is the task completion time
        dddN = np.zeros((self.BSpline_num, len(alpha)))
        self.BSjerk = np.zeros((3, len(alpha)))

        for j in range(len(alpha)):
            for i in range(self.BSpline_num):#Sigma^{BSnum-1}_{i=0} P_i * b^t_{i,k}(alpha(j))
                if j==len(alpha)-1:
                    dddN[i, j] = self.db(3, i, self.BSpline_deg, self.knot[-2])
                else:
                    dddN[i, j] = self.db(3, i, self.BSpline_deg, alpha[j])
                self.BSjerk[0, j] += dddN[i, j]*self.BSpline_P[i][0]
                self.BSjerk[1, j] += dddN[i, j]*self.BSpline_P[i][1]
                self.BSjerk[2, j] += dddN[i, j]*self.BSpline_P[i][2]        

if __name__=="__main__":
    planner = BSplinePlanner()
    #define control point
    P = np.array([[3,3,0], [2.9,2.9,0], [9,1,0], [9,5,0], [12,5,0], [12,8,0], [11.9,0.1,0], [12,0,0]])
    #path planner Bspline set parameters
    #degree must be larger than 2
    planner.setBSplineParam(3, P)
    #make time parameter alpha
    alpha = np.linspace(0, 1, 100)
    #path planning
    planner.BSpline(alpha)
    planner.dBSpline(alpha)
    planner.ddBSpline(alpha)
    planner.dddBSpline(alpha)

       #微分が正しいか計算
    dx = np.zeros(len(alpha))
    dy = np.zeros(len(alpha))
    dz = np.zeros(len(alpha))
    for i in range(1, len(alpha)):
        dx[i] = (planner.BSpoint[0, i] - planner.BSpoint[0, i-1])/(alpha[i]-alpha[i-1])
        dy[i] = (planner.BSpoint[1, i] - planner.BSpoint[1, i-1])/(alpha[i]-alpha[i-1])
        dz[i] = (planner.BSpoint[2, i] - planner.BSpoint[2, i-1])/(alpha[i]-alpha[i-1])
    plt.plot(alpha, planner.BSvel[0, :], label='fomula')
    plt.plot(alpha[1:], dx[1:], label='nomal')
    plt.legend()
    plt.show()

    #加速度が正しいか計算
    ddx = np.zeros(len(alpha))
    ddy = np.zeros(len(alpha))
    ddz = np.zeros(len(alpha))
    for i in range(2, len(alpha)):
        ddx[i] = (dx[i] - dx[i-1])/(alpha[i]-alpha[i-1])
    plt.plot(alpha, planner.BSacc[0, :], label='formula')
    plt.plot(alpha[2:], ddx[2:], label='normal')
    plt.legend
    plt.show()
    
    plt.plot(alpha, planner.BSjerk[0, :])
    plt.show()