# Plusieurs facteurs peuvent influencer le prix d'une option sur action
# Le cours du sous-jacent : S0 > 0
# Le prix d'exercice (strike) : K
# le taux d'intérêt sans risque : r
# la volatilité du prix de l'action : sigma
# le temps restant à l'échéance : T (en années) >= t >= 0

from http.cookies import CookieError
from numpy import zeros, linspace, exp, maximum, delete, diag, linalg

# Initialisation des paramètres financiers :
S = 50 # cours du sous-jacent
K = 50 # strike
r = 0.05 # taux d'intérêt sans risque
sigma = 0.2 # volatilité
T = 3  # maturité en années

# Initialisation des paramètres numériques :
N = 200 # nombre de de points de maillage en temps
M = 150 # nombre de de points de maillage d'actifs
S0 = 0 # valeur extreme minimale du sous-jacent
Smax = 150 # valeur extreme maximale du sous-jacent

# Initialisation du maillage et de la matrice du système linéaire :
solution_mesh =  zeros((N,M)) # tableau de la solution approchée
Smesh = linspace(0,Smax, M).astype(int) # maillage en S
Tmesh = linspace(T,0,N) # maillage en t
dt = T/N # pas de temps
for i in range(M):
    solution_mesh[0,i] = maximum(K-Smesh[i],0) # condition initiale
for i in range(N):
    solution_mesh[i,0] = K*exp(-r*(T-Tmesh[i])) # condition aux limites en S=0
    solution_mesh[i,M-1] = 0 # condition aux limites en S=M

# Définition des fonctions pour résoudre le système linéaire :
def A(i):
    return 0.5*dt*(r*i - sigma**2*i**2)
def B(i):
    return 1 + dt*(sigma**2*i**2 + r)
def C(i):
    return -0.5*dt*(sigma**2*i**2 + r*i)

# Construction de la matrice tri-diagonale et de son inverse :
Acoeffs = zeros((M+1,1))
Bcoeffs = zeros((M+1,1))
Ccoeffs = zeros((M+1,1))

for i in range(1,M):
    Acoeffs[i] = A(i)
    Bcoeffs[i] = B(i)
    Ccoeffs[i] = C(i)

Acoeffs_prepend = delete(Acoeffs, 0)
Ccoeffs_prepend = delete(Ccoeffs, len(Ccoeffs)-1)


Tri = diag(Acoeffs_prepend,-1) + diag(Bcoeffs) + diag(Ccoeffs_prepend,1)
print(linalg.det(Tri))
# Tri_inv = linalg.inv(Tri)
