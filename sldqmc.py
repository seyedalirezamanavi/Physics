import numpy as np
import time
from scipy import linalg as LA
from numba import jit
import os
import errno


class DQMC():
    # This class is difined to initialize the Determinental Quantum monte carlo
    def __init__(self, t_x, t_y, t_xy, mu, U, DT, beta, Nx, Ny, bnd_x, bnd_y):
        self.t_x = t_x  # hopping in x direction
        self.t_y = t_y  # hopping in y direction
        self.t_xy = t_xy  # hopping in xy(diagnal) direction
        self.mu = mu  # mu, number conservation coeficient
        self.U = U  # U, interaction coeficient
        self.Nx = Nx  # Number of particles along x direction
        self.Ny = Ny  # Number of particles along y direction
        self.bnd_x = bnd_x  # boundary condition along x direction; 1 to be periodic. 0 to be open.
        self.bnd_y = bnd_y  # boundary condition along y direction; 1 to be periodic. 0 to be open.
        self.DT = DT  # DT, kinetic matrix coeficient
        self.L = int(beta/self.DT)  # L, lengh of lattice along time direction.
        self.beta = beta  # inverse tempreture.
        self.lamda = np.arccosh(np.exp(self.U*self.DT/2))  # interaction matrix coeficient
        self.Ns = Nx*Ny  # over-all number of particles

    def kinetic(self):
        # Constructing the matrix of hopping amplitudes for a 2D square lattice
        dx = 1  # range of hopping along x
        rx_max = dx  # range of hopping along x
        if self.Nx == 1:
            rx_max = 0
        elif self.Nx < 5:
            rx_max = 1

        dy = 1  # range of hopping along y
        ry_max = dx  # range of hopping along y
        if self.Ny == 1:
            ry_max = 0
        elif self.Ny < 5:
            ry_max = 1

        # hopping amplitudes from sit i to its nearest and next nearest neighbors
        T = [[self.t_xy, self.t_x, self.t_xy],
             [self.t_y, 0, self.t_y],
             [self.t_xy, self.t_x, self.t_xy]]
        T = np.dot(-1, T)
        # hopping matrix
        H0 = np.zeros((self.Nx*self.Ny, self.Nx*self.Ny))

        for i1 in range(self.Nx):
            for i2 in range(self.Ny):

                r1 = i1*self.Ny + i2  # index of site r1 = (i1,i2)
                for rx in range(-rx_max, rx_max+1):
                    j1 = i1 + rx
                    if self.bnd_x == 1:
                        j1 = np.mod(j1, self.Nx)
                    for ry in range(-ry_max, ry_max+1):
                        j2 = i2 + ry
                        if self.bnd_y == 1:
                            j2 = np.mod(j2, self.Ny)

                        r2 = j1*self.Ny + j2  # index of site r2 = (j1,j2)
                        if j1 >= 0 and j1 < self.Nx:
                            if j2 >= 0 and j2 < self.Ny:

                                H0[r1, r2] = T[rx+dx, ry+dy]
                                H0[r2, r1] = H0[r1, r2]

        H0 = (H0 + H0.transpose())/2  # symmetrize hopping matrix

        H0 += np.eye(len(H0)) * (-self.mu)  # adding mu term

        H0 = np.dot(self.DT, H0)

        return H0

    def interaction(self):
        # Constructing interaction matrix of randomly +-1 corresponding to up and down spins
        v = np.dot(self.lamda, np.power(-1, np.random.randint(2, size=(self.L, self.Nx*self.Ny))))
        return v


@jit(nopython=True, fastmath=True)
def qr_decomposition(M_i, T0):
    # qr decomposition for numerical stability of matrix productions
    Q, R = np.linalg.qr(M_i)
    D0 = np.diag(R)
    inv_D = np.diag(np.divide(np.ones(np.shape(D0)), D0))
    D = np.diag(D0)
    T = np.dot(np.dot(inv_D, R), T0)
    return Q, D, T


def lu_decomposition(A1, D_b, A3):
    # lu decomposition for stabilization log(det())
    __, L1, U1 = LA.lu(A1)
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

    sign_det_G = sign_det_A1*sign_det_A2*sign_det_A3  # sign of det(G)
    log_det_G = -log_det_A1+log_det_A2+log_det_A3  # log of abs(det(G))

    return log_det_G, sign_det_G


@jit(nopython=True)
def partition_sigma(v, expmk, alpha, p_vec):
    # calculate C = (B_{l}B_{l-1}...B_{2}B_{1}) concerning qr stabilization
    Q = np.eye(np.shape(expmk)[0])
    D = np.eye(np.shape(expmk)[0])
    T = np.eye(np.shape(expmk)[0])

    i = 0
    while i < len(v):
        a0 = alpha*v[i]
        ddd = np.diag(np.exp(a0))
        B = expmk.dot(ddd)
        i += 1
        while i < len(v) and p_vec[i] < 1:  # qr decompose after k matrix multiplication.
            a1 = alpha*v[i]
            ddd1 = np.diag(np.exp(a1))
            b = expmk.dot(ddd1)
            B = b.dot(B)
            i += 1

        C = B.dot(Q)
        C = C.dot(D)
        Q, D, T = qr_decomposition(C, T)  # decomposition C to orthogonal matrices Q, D, T.

    return Q, D, T


def initialize_green_function(expmk, v, alpha, p_vec):
    # initialize greens matrix using Q, D and T.
    Q, D, T = partition_sigma(v, expmk, alpha, p_vec)

    D_diag = np.diag(D)

    # divide D to D_b for large value of D and D_s for small value of D
    D_b = np.multiply(np.maximum(np.ones(np.shape(D_diag)), np.abs(D_diag)), np.sign(D_diag))
    D_s = np.minimum(np.ones(np.shape(D_diag)), np.abs(D_diag))
    inv_D_b = np.diag(np.divide(1, D_b))
    D_s = np.diag(D_s)

    A1 = np.add(np.matmul(inv_D_b, (Q.transpose())), np.matmul(D_s, T))
    A2 = inv_D_b
    A3 = Q.transpose()

    G = np.matmul(LA.solve(A1, A2), A3)
    log_det_G, sign_det_G = lu_decomposition(A1, D_b, A3)  # using LU decomposition to calculate log_det_G and the sign value
    return G, log_det_G, sign_det_G


def update_Greens(expmk, v, l, p_vec):
    # update greens matrix with sherman-morison algorithm
    Gu, log_det_G_up, sign_det_G_up = initialize_green_function(expmk, v, 1, p_vec)
    Gd, log_det_G_dn, sign_det_G_dn = initialize_green_function(expmk, v, -1, p_vec)
    I = np.eye(hubbard.Ns)
    for n in range(hubbard.Ns):
        s_new = v[0, n]
        alpha_up = np.exp(-2*s_new)-1
        du = 1+(1-Gu[n, n])*(alpha_up)
        alpha_dn = np.exp(2*s_new)-1
        dd = 1+(1-Gd[n, n])*(alpha_dn)
        d = dd * du

        # local update using metropolis-hastings algorithm
        r = np.random.random()
        if np.abs(d) > r:

            a_up = np.subtract(I, Gu)
            B_up = Gu
            Gu = np.subtract(Gu, np.dot(alpha_up/du, np.matmul(np.array([a_up[:, n]]).T, np.array([B_up[n, :]]))))

            a_dn = np.subtract(I, Gd)
            B_dn = Gd
            Gd = np.subtract(Gd, np.dot(alpha_dn/dd, np.matmul(np.array([a_dn[:, n]]).T, np.array([B_dn[n, :]]))))

            v[0, n] = -s_new  # flip corresponding spin if the change has accepted
        else:
            pass
            # print("%f rejected"%d)

    return Gu, Gd, v


def logpartition(expmk, v, p_vec):
    # calculate log(det(G)) where its needed
    Gu, log_det_G_up, sign_det_G_up = initialize_green_function(expmk, v, 1, p_vec)
    Gd, log_det_G_dn, sign_det_G_dn = initialize_green_function(expmk, v, -1, p_vec)

    return log_det_G_up + log_det_G_dn


def wrap(Gu, Gd, v, expmk, expmmk, p_vec, l):
    # time wraping after space update
    v = np.roll(v, -1, axis=0)  # shift hubbard-stratonovich field -1 along time
    p_vec = np.roll(p_vec, -1, axis=0)  # shift the p_vec -1 step

    # calculate effective hamiltonian from the scratch every time p_vec is one
    if p_vec[l] == 1:
        Gu, log_det_G_up, sign_det_G_up = initialize_green_function(expmk, v, 1, p_vec)
        Gd, log_det_G_dn, sign_det_G_dn = initialize_green_function(expmk, v, -1, p_vec)
        Heff = (log_det_G_dn + log_det_G_up)/hubbard.beta

    # wrap greens matrix
    else:
        v_tmp = v[hubbard.L-1, :]
        B = expmk
        inv_B = expmmk
        B_up_prev = np.dot(B, LA.expm(np.diag(v_tmp)))
        inv_B_up_prev = np.dot(LA.expm(np.diag(-v_tmp)), inv_B)
        Gu = np.matmul(B_up_prev, np.matmul(Gu, inv_B_up_prev))

        B_dn_prev = np.matmul(B, LA.expm(np.diag(-v_tmp)))
        inv_B_dn_prev = np.matmul(LA.expm(np.diag(v_tmp)), inv_B)
        Gd = np.matmul(B_dn_prev, np.matmul(Gd, inv_B_dn_prev))

        Heff = logpartition(expmk, v, p_vec)/hubbard.beta

    return Gu, Gd, v, p_vec, Heff


def conf(v):
    # Construct 1D space-time spin field out of hubbard-stratonovich field by reshapping.
    s = v[0]
    for i in range(len(v)-1):
        s = np.concatenate((s, v[i+1]))
    s = np.array([s])
    return s


# initialize Determinental Quantum monte carlo parameters
hubbard = DQMC(t_x=1, t_y=1, t_xy=0, mu=0, U=4, DT=1/8,
               beta=5, Nx=10, Ny=1, bnd_x=1, bnd_y=1)
k = hubbard.kinetic()  # initialize kinetic matrix
expmk = np.abs(LA.expm(-k))  # expm(k)
expmmk = LA.expm(k)  # expm(-k)

results = ("results-Beta%d-Nx%d-Ny%d-Mu%d-U%d-DT%f-boundary condition(%d,%d)-tx%d"%(hubbard.beta, hubbard.Nx, hubbard.Ny, hubbard.mu, hubbard.U, hubbard.DT, hubbard.bnd_x, hubbard.bnd_y, hubbard.t_x))

try:
    os.mkdir(results)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# generating p_vec: every k zeros there exists a 1.
p_vec = np.zeros(hubbard.L)
for i in range(10, hubbard.L, 10):
    p_vec[i] = 1
p_vec[-1] = 1


iterations = 500  # monte carlo iterations
l_measurment = int(0.2*iterations)  # do measurments after 20% of iterations passed
markov = 200

for i in range(markov):
    v = hubbard.interaction()  # initialize interaction matrix
    # np.load("v.npy")
    t = time.time()
    fosav = []
    for msr in range(iterations):
        t0 = time.time()
        for l in range(hubbard.L):
            
            Gu, Gd, v = update_Greens(expmk, v, l, p_vec)   # sherman-morison
            Gu, Gd, v, p_vec, Heff = wrap(Gu, Gd, v, expmk, expmmk, p_vec, l)  # time wrap

        if msr > l_measurment:
            fosav.append([Heff, v])  # effective hamiltonian and configuration for save

        print("Running msr %d : %f s" % (msr, time.time()-t0))

    print("Running markov %d : %f s" % (i, time.time()-t))

    np.save(results+"/"+str(int(time.time())), fosav)
