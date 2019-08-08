import numpy as np
import matplotlib.pyplot as plt
# importing methods
from initialize_k_V import Kinetic, Interaction
from Greens import initialize_green_function, update, wrapping
# initialize problems parameters
U = 1
t = 1
DT = .1
N = 20
beta = 10
L = int(beta/DT)
lamda = np.arccosh(np.exp(U*DT/2))
mu = 0
BC = 0
# Initialize Kinetic Energy Matrix and Interaction Energy Matrix
k = Kinetic(N, BC, t, mu)
v = Interaction(N, L, lamda)


for l in range(L-1,0,-1):
    for ti in range(10):
        tt = np.random.randint(N)
        Gu , Gd , v = update(k,v,tt,l,N)
    Gu = wrapping(k,v,Gu,l)
    Gd = wrapping(k,v,Gd,l)
