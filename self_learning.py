import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from dqmc import initialize_green_function, DQMC, conf


def H(beta, S, expmk, p_vec):
    # calculate the exact energy of configuration using DQMC
    v = np.reshape(S, (hubbard.L, hubbard.Ns))
    Gu, log_det_G_up, sign_det_G_up = initialize_green_function(expmk, v, 1, p_vec)
    Gd, log_det_G_dn, sign_det_G_dn = initialize_green_function(expmk, v, -1, p_vec)

    return beta*(log_det_G_dn+log_det_G_up), Gu+Gd


def Heff(S, J):
    # calculate the two point effective hamiltonian of given configuration
    return np.matmul(S, np.matmul(J, S.T))


def local_update(beta, S, J):
    # hubbard-stratonovich field update using metropolis-hastings algorithm
    S = np.array(S)
    for i in range(len(S)):
        Hef = Heff(S, J)
        Sp = S.copy()
        Sp[i] = -Sp[i]
        Hefp = Heff(Sp, J)
        r = -beta*Hefp+beta*Hef
        if r > np.log(np.random.random()):
            S = Sp.copy()
#             print(np.abs(np.log(np.random.random())),np.abs(r))
#             print("accepted",r)
#         else:
#             print("rejected",r)
#             print(np.abs(np.log(np.random.random())),np.abs(r))
    return S


def cummulative_update(beta, S, Sp, J, expmk, p_vec):
    # the cummulative update
    G = 0
    Hef = Heff(S, J)
    Hefp = Heff(Sp, J)
    Ha, Gr = H(beta, S, expmk, p_vec)
    Hap, Grp = H(beta, Sp, expmk, p_vec)
    r = -(Hap - Ha)-(-beta*Hef+beta*Hefp)
    print(r)
    if r > np.log(np.random.random()):
        print("a")
        S = Sp.copy()
        G = Grp.copy()
    return S, G, r


hubbard = DQMC(t_x=1, t_y=1, t_xy=0, mu=0, U=4, DT=1/8, beta=5, Nx=10, Ny=1, bnd_x=1, bnd_y=1)
k = hubbard.kinetic()
expmk = np.abs(LA.expm(-k))
expmkk = np.abs(LA.expm(k))
p_vec = np.zeros(hubbard.L)
for i in range(10, hubbard.L, 10):
    p_vec[i] = 1
p_vec[-1] = 1

J = np.load("J70000.npy")


Gl = []
for i in range(200):
    v = hubbard.interaction()
    S = conf(v)[0]
    r = 1
    while r > 0:
        Sp = local_update(hubbard.beta, S, J)
        S, G, r = cummulative_update(hubbard.beta, S, Sp, J, expmk, p_vec)
    Gl.append(G)


ZZ = 0
for G in Gl:
    ZZ += np.diag(expmkk.dot(G).dot(expmk))

plt.plot(ZZ/np.shape(Gl)[0])
plt.show()
