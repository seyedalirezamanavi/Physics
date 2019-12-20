import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy import linalg as LA
from scipy.sparse import spdiags
from numba import jit,objmode
import os

def functimer(func):

    def wrapper(*args,**kwargs):
        t0=time.time()
        result=func(*args,**kwargs)
        #print("Running %s : %f s"%(func.__name__,time.time()-t0))
        return result
    return wrapper


class SLDQMC():
    
    #@functimer
    def __init__(self,t_x,t_y,t_xy,mu,U,DT,beta,Nx,Ny,bnd_x,bnd_y):
        self.t_x=t_x
        self.t_y=t_y
        self.t_xy=t_xy
        self.mu=mu
        self.U=U
        self.Nx=Nx
        self.Ny=Ny
        self.bnd_x = bnd_x
        self.bnd_y = bnd_y
        self.DT = DT#np.sqrt(1/(10*U*np.sqrt(t_x**2+t_y**2)))
        self.L = int(beta/self.DT)
        self.beta = beta
        self.lamda = np.arccosh(np.exp(self.U*self.DT/2))
        self.Ns = Nx*Ny
        
    #@functimer
    def kinetic(self):
        #Constructing the matrix of hopping amplitudes for a 2D square lattice
        dx = 1 # range of hopping along x
        rx_max = dx # range of hopping along x
        if self.Nx == 1:
            rx_max = 0
        elif self.Nx < 5:
            rx_max = 1
        
        dy = 1 #range of hopping along y
        ry_max = dx # range of hopping along y
        if self.Ny == 1:
            ry_max = 0
        elif self.Ny < 5:
            ry_max = 1
        

        # hopping amplitudes from sit i to its nearest and next nearest neighbors
        T = [[self.t_xy,self.t_x,self.t_xy],
            [ self.t_y ,0  ,self.t_y ],
            [ self.t_xy,self.t_x,self.t_xy]]
        T = np.dot(-1,T)
        # hopping matrix
        H0 = np.zeros((self.Nx*self.Ny,self.Nx*self.Ny))
        
        for i1 in range(self.Nx):
            for i2 in range ( self.Ny ):
                
                r1 = i1*self.Ny + i2 # index of site r1 = (i1,i2)
                for rx in range(-rx_max,rx_max+1):
                    j1 = i1 + rx
                    if self.bnd_x == 1:
                        j1 = np.mod(j1,self.Nx)
                    for ry in range(-ry_max,ry_max+1):
                        j2 = i2 + ry
                        if self.bnd_y == 1:
                            j2 = np.mod(j2,self.Ny)
                        

                        r2 = j1*self.Ny + j2 # index of site r2 = (j1,j2)
                        if j1>=0 and j1 < self.Nx:
                            if j2 >=0 and j2 < self.Ny:
                                #print(r1,r2)
                                H0[r1,r2] = T[rx+dx,ry+dy]
                                H0[r2,r1] = H0[r1,r2]
                            
                        
        H0 = (H0 + H0.transpose())/2
        
        H0 += np.eye(len(H0)) * (-self.mu)
        
        H0 = np.dot(self.DT,H0) 
        #print(H0)
        #plt.imshow(H0)
        #plt.show()
        return H0
        
    #@functimer
    def interaction(self):
        v = np.dot(self.lamda,np.power(-1,np.random.randint(2,size=(self.L,self.Nx*self.Ny)) ))
        return v





@jit(nopython=True,fastmath = True)    
def qr_decomposition(M_i,T0):
    #print("mi",M_i)
    Q,R = np.linalg.qr(M_i)

    D0 = np.diag(R)

    #print("R",R)

    #print("D0",D0)

    inv_D = np.diag(np.divide(np.ones(np.shape(D0)),D0))

    #print("inv_D",inv_D)

    D = np.diag(D0)

    

    T = np.dot(np.dot(inv_D,R),T0)
        
    #print("T",T)

    

    return Q,D,T
    

#@jit(nopython=True)
def lu_decomposition(A1,D_b,A3):
    aaa,L1,U1 = LA.lu(A1)

    log_det_L1 = 0
    sign_det_L1 = np.sign(LA.det(L1))
    log_det_U1 = np.sum(np.log(np.abs(np.diag(U1))))
    sign_det_U1 = np.prod(np.sign(np.diag(U1)))
    log_det_A1 = log_det_L1 + log_det_U1
    sign_det_A1 = sign_det_L1*sign_det_U1

    log_det_A2 = -np.sum(np.log(np.abs(D_b)))
    sign_det_A2 = np.prod(np.sign(D_b))

    log_det_A3 = 0
    sign_det_A3 = np.sign(LA.det(A3))

    sign_det_G = sign_det_A1*sign_det_A2*sign_det_A3 # sign of det(G)
    log_det_G = -log_det_A1+log_det_A2+log_det_A3 # log of abs(det(G))
    #print("log_det_G",log_det_G)
    return log_det_G,sign_det_G
    
@jit(nopython=True) 
def partition_sigma(v,expmk,alpha,p_vec):

    Q = np.eye(np.shape(expmk)[0])
    D = np.eye(np.shape(expmk)[0])
    T = np.eye(np.shape(expmk)[0])
    
    
    i = 0
    while i < len(v):
        a0 = alpha*v[i]
        ddd = np.diag(np.exp(a0))
        B = expmk.dot(ddd)
        i+=1
        while i < len(v) and p_vec[i]<1:
            a1 = alpha*v[i]
            ddd1 = np.diag(np.exp(a1))
            b = expmk.dot(ddd1)
            B = b.dot(B)
            i+=1
            #print("b",b)
            #print("B",B)

        C = B.dot(Q)
        C = C.dot(D)
        #print(C.dtype,B.dtype,Q.dtype,D.dtype)
        Q, D, T = qr_decomposition(C,T)
        #plt.imshow(T)
        #plt.show()
    return Q, D, T




#@functimer
def initialize_green_function(expmk,v,alpha,p_vec):  
    Q, D, T = partition_sigma(v,expmk,alpha,p_vec)
    #plt.imshow(np.matmul(Q,D))
    #plt.show()
    #print(D)
    D_diag = np.diag(D)
    D_b = np.multiply(np.maximum(np.ones(np.shape(D_diag)),np.abs(D_diag)),np.sign(D_diag))
    D_s = np.minimum(np.ones(np.shape(D_diag)),np.abs(D_diag)) 

    inv_D_b = np.diag(np.divide(1,D_b))
    D_s = np.diag(D_s)
    
    A1 = np.add(np.matmul(inv_D_b,(Q.transpose())) ,np.matmul( D_s , T))
    A2 = inv_D_b
    A3 = Q.transpose()
    #print("A1",A1)
    #print("A2",A2)
    #print("A3",A3)
    G = np.matmul(LA.solve(A1,A2),A3)
    #print("G",G)
    #plt.imshow(G)
    #plt.show()

    #print(LA.solve(A1,A2))
    
    log_det_G,sign_det_G = lu_decomposition(A1,D_b,A3)
    #print(log_det_G,sign_det_G)
    #plt.imshow(G)
    #plt.show()
    return G,log_det_G,sign_det_G

#@functimer
#@jit(nopython=True) 
def update_Greens(expmk,v,l,p_vec):
    Gu,log_det_G_up,sign_det_G_up = initialize_green_function(expmk,v,1,p_vec)
    Gd,log_det_G_dn,sign_det_G_dn = initialize_green_function(expmk,v,-1,p_vec)
    #print(l,n,Gu)
    #plt.imshow(Gu)
    #plt.show()
    
    I = np.eye(hubbard.Ns)
    for n in range(hubbard.Ns):
        #print(n)
        s_new =  v[0,n]
        alpha_up = np.exp(-2*s_new)-1
        du = 1+(1-Gu[n,n])*(alpha_up)
        alpha_dn = np.exp(2*s_new)-1
        dd = 1+(1-Gd[n,n])*(alpha_dn)
        d = dd * du
        #print("d",d)
        
        #print(d)
        r = np.random.random()
        if np.abs(d) > r :
            #print("%f accepted"%d)
            a_up = np.subtract(I,Gu)
            B_up = Gu
            Gu = np.subtract(Gu,np.dot(alpha_up/du,np.matmul(np.array([a_up[:,n]]).T,np.array([B_up[n,:]]))))
            
            a_dn = np.subtract(I,Gd)
            B_dn = Gd
            Gd = np.subtract(Gd,np.dot(alpha_dn/dd,np.matmul(np.array([a_dn[:,n]]).T,np.array([B_dn[n,:]]))))
            
            #plt.imshow(Gu)
            #plt.show()
            v[0,n] = -s_new
            #print("pass")
        else:
            pass
            #print("%f rejected"%d)
    #plt.imshow(v)
    #plt.show()
    return Gu, Gd, v
    


def logpartition(expmk,v,p_vec):
    Gu,log_det_G_up,sign_det_G_up = initialize_green_function(expmk,v,1,p_vec)
    Gd,log_det_G_dn,sign_det_G_dn = initialize_green_function(expmk,v,-1,p_vec)
        
    return log_det_G_up + log_det_G_dn


#@jit(nopython=True) 
def wrap(Gu,Gd,v,expmk,expmmk,p_vec,l):
    #plt.imshow(v)
    #plt.show()
    v = np.roll(v,-1,axis = 0)
    p_vec = np.roll(p_vec,-1,axis = 0)
    
    #print(p_vec)
    if p_vec[l]==1:
        Gu,log_det_G_up,sign_det_G_up = initialize_green_function(expmk,v,1,p_vec)
        Gd,log_det_G_dn,sign_det_G_dn = initialize_green_function(expmk,v,-1,p_vec)
        Heff = (log_det_G_dn + log_det_G_up)/hubbard.beta
        
    else:
        
        v_tmp = v[hubbard.L-1,:]
        #print(v_tmp)
        
        B = expmk
        inv_B = expmmk
        #print("B",B )
        #print("inv_B",inv_B)
        B_up_prev = np.dot(B,LA.expm(np.diag(v_tmp)))
        #print("B_up_prev",B_up_prev)
        inv_B_up_prev = np.dot(LA.expm(np.diag(-v_tmp)),inv_B)
        #print("inv_B_up_prev",inv_B_up_prev)
        Gu = np.matmul(B_up_prev,np.matmul(Gu,inv_B_up_prev))
        #print("Gu",Gu)
        
        B_dn_prev = np.matmul(B,LA.expm(np.diag(-v_tmp)))
        inv_B_dn_prev = np.matmul(LA.expm(np.diag(v_tmp)),inv_B)
        Gd = np.matmul(B_dn_prev,np.matmul(Gd,inv_B_dn_prev))
        #print("Gd",Gd)
        
        
        Heff = logpartition(expmk,v,p_vec)/hubbard.beta
    
    #plt.imshow(Gu)
    #plt.show()
    return Gu,Gd,v,p_vec,Heff
        

def conf(v):
    s = v[0]
    for i in range(len(v)-1):
        s = np.concatenate((s,v[i+1]))
    s = np.array([s])
    return s



        
        
     
     
     
hubbard = SLDQMC(t_x=1,t_y=1,t_xy=0,mu=0,U=4,DT = 1/8 ,beta=5,Nx=10, Ny=1, bnd_x=1,bnd_y=1)
k = hubbard.kinetic()
expmk = np.abs(LA.expm(-k))
expmmk = LA.expm(k)



results = ("results-Beta%d-Nx%d-Ny%d-Mu%d-U%d-DT%f-boundary condition(%d,%d)-tx%d"%(hubbard.beta,hubbard.Nx,hubbard.Ny,hubbard.mu,hubbard.U,hubbard.DT,hubbard.bnd_x,hubbard.bnd_y,hubbard.t_x))
print(results)
try:
    os.mkdir(results)
except:
    print("path existed")

p_vec=np.zeros(hubbard.L)
for i in range(10,hubbard.L,10):
    p_vec[i]=1 
p_vec[-1] = 1



monte = 500
l_measurment = int(0.2*monte)
for i in range(200):
    v = hubbard.interaction()
    #np.load("v.npy")
    t=time.time()
    fosav = []
    for msr in range(monte):
        t0=time.time()
        for l in range(hubbard.L):
            Gu , Gd , v = update_Greens(expmk,v,l,p_vec)

            Gu,Gd,v,p_vec,Heff = wrap(Gu,Gd,v,expmk,expmmk,p_vec,l)
      
        if msr>l_measurment:
            fosav.append([Heff,v])
          
        print("Running msr %d : %f s"%(msr,time.time()-t0))
       
    print("Running mark %d : %f s"%(i,time.time()-t))
    
    
    

    np.save(results+"/"+str(int(time.time())),fosav)



     
     
     
     
     
     
     
     

