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