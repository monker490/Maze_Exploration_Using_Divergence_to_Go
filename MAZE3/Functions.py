import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

# N = 8

# xu = [0,0]
# yu = [5,5]
# cov = [[1,0],[0,1]]

# x = np.random.multivariate_normal(xu, cov, N)
# y = np.random.multivariate_normal(yu, cov, N)
# x = np.reshape(np.array([[0,0],[0,1],[0,2],[1,0]]),(4,2))
# y = np.reshape(np.array([[1,2],[2,0],[2,1],[2,2]]),(4,2))

def entropy(g,sig):
    Hx = []
    Ix = []
    K = len(g)
    var = 2*(sig**2)
    cov = var*np.identity(2) 
    det = np.linalg.det(cov)
    cov = np.linalg.inv(cov)
    # print(cov)
    for c,i in enumerate(g):
        for d,j in enumerate(g):
            # if (d!=c):
            num = (j - i)
            num = np.reshape(num,(1,2))
            pwr = -0.5*np.matmul(np.matmul(num,cov),num.T)
            term = (1/(np.sqrt(det)*2*np.pi))*(np.exp(pwr))
            Hx += [np.squeeze(term).tolist()]
            Ix += [np.squeeze(np.matmul(num,num.T)).tolist()]

    # print (np.shape(Hx))
    mean = np.reshape(np.array([np.mean(Hx)]),(1,-1))    
    Hx = np.reshape(np.array([-np.log(np.sum(Hx)/K**2)]),(1,-1))
    # print(mean,Hx)
    return (Hx,mean)



def divergence(x,y,sigx,sigy,width):
    N = len(x)
    varx = 2*(sigx**2)
    vary = 2*(sigy**2)
    cov = np.array([[varx,0],[0,vary]])
    # xsim = rbf_kernel(x,x,gamma = 1/(width)**2)
    # ysim = rbf_kernel(y,y,gamma = 1/(width)**2)
    xsim = rbf_kernel(x,x,gamma = 1/varx)
    ysim = rbf_kernel(y,y,gamma = 1/vary)
    xsim /= np.sum(xsim)
    ysim /= np.sum(ysim)
    det = np.sqrt(varx*vary)
    H = []
    for c,i in enumerate(x):
        mu = y[c] - x[c]
        temp = []
        for j in y:
            k = (j - i - mu)
            k = np.reshape(k,(1,2))
            pwr = -0.5*np.matmul(np.matmul(k,cov),k.T)
            term = (1/(np.sqrt(det)*2*np.pi))*(np.exp(pwr))
            temp += [np.squeeze(term).tolist()]
        H += [temp]
    
    Vx = np.matmul(np.matmul(xsim.T,H),xsim)
    Vx = np.sum(Vx)
    Vx = Vx/N**2

    Vy = np.matmul(np.matmul(ysim.T,H),ysim)
    Vy = np.sum(Vy)
    Vy = Vy/N**2

    Vc = np.matmul(np.matmul(xsim.T,H),ysim)
    Vc = np.sum(Vc)
    Vc = Vc/N**2

    # print("Vx: ", Vx)
    # print("Vy: ", Vy)
    # print("Vc: ", Vc)

    Deuc = Vx + Vy - 2*Vc
    Dcs = np.log((Vx*Vy)/(Vc**2))

    # print("Deuc: ", Deuc)
    # print("Dcs: ", Dcs)
    return(Deuc,Dcs)

# print(x)
# print(y)
# print("hi")
# a,b = divergence(x,y,0.01,0.05,1)
# print(a,b)
