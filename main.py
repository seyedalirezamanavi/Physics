import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import linalg as LA
from sldqmc import SLDQMC,functimer
import os 


hubbard = SLDQMC(t_x=1,t_y=1,t_xy=0,mu=0,U=4,DT = 1/8 ,beta=5,Nx=10, Ny=1, bnd_x=1,bnd_y=1)
k = hubbard.kinetic()
expmk = np.abs(LA.expm(-k))
expmmk = LA.expm(k)

zl = []
cl = []

results = ("results-Beta%d-Nx%d-Ny%d-Mu%d-U%d-DT%f-boundary condition(%d,%d)-tx%d.npy"%(hubbard.beta,hubbard.Nx,hubbard.Ny,hubbard.mu,hubbard.U,hubbard.DT,hubbard.bnd_x,hubbard.bnd_y,hubbard.t_x))
print(results)
try:
    os.mkdir(results)
except:
    print("path existed")

p_vec=np.zeros(hubbard.L)
for i in range(10,hubbard.L,10):
    p_vec[i]=1 
p_vec[-1] = 1

fosav = []
rr = []

monte = 30
l_measurment = int(0.2*monte)
for i in range(3):
    v = hubbard.interaction()
    t=time.time()
    for msr in range(monte):
        t0=time.time()
        for l in range(hubbard.L):
            Gu , Gd , v = hubbard.update_Greens(expmk,v,l,p_vec)
            #print("vsh",v)
            #print("gush",Gu)
            #plt.imshow(Gu)
            #plt.show()
            
            Gu,Gd,v,p_vec,Heff = hubbard.wrap(Gu,Gd,v,expmk,expmmk,p_vec,l)
            #print("Gw",Gu)
            #print("vw",v)
            #plt.imshow(Gu)
            #plt.show()
            #print("Heff",Heff)
            c = hubbard.conf(v)
        if msr>l_measurment:
            fosav.append([Heff,v])
            cl.append(c)
            zl.append(Heff)
            rr.append(hubbard.JJ(v,Heff))
        #print(Heff,i,msr)
        
        print("Running msr %d : %f s"%(msr,time.time()-t0))
        #plt.imshow(Gu+Gd)
        #plt.show()
    print("Running mark %d : %f s"%(i,time.time()-t))
    
    
    
rr = np.array(rr)
print(rr.shape)
plt.plot(rr[...,1],rr[...,0],'.')
plt.show()
np.save(results+"/"+str(int(time.time())),fosav)

con = sorted(zip(zl,cl),reverse = True)

#plt.imshow(hubbard.effectiveJmonte(v,con))
#plt.show()
J, M= hubbard.effectiveJ(v,con)
print(J)
plt.imshow(J)
plt.show()
plt.figure()
plt.plot(M[1:])
plt.show()
plt.figure()
plt.plot(J[0][1:])
plt.show()
r = np.reshape(J[0][:],(hubbard.L,hubbard.Ns))
print(r)
plt.plot(r[0,1:])
plt.show()
plt.imshow(r)
plt.show()
plt.plot(np.sort(zl),".")

plt.show()



#cc = np.matmul(np.matrix(c).transpose(),np.matrix(c))
#print(cc)
#print(LA.det(cc))
