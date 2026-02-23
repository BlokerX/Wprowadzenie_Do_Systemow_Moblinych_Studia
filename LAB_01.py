import math
import matplotlib.pyplot as plt

# ---- Generator U(0,1) (LCG) ----
class LCG:
    def __init__(self, seed=1):
        self.m = 2**31 - 1
        self.a = 1103515245
        self.c = 12345
        self.state = seed

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

# ---- Poisson (Knuth) ----
def poisson(l, gen):
    L = math.exp(-l)
    k, p = 0, 1
    while p > L:
        k += 1
        p *= gen.rand()
    return k - 1

# ---- Normalny (Box-Muller) ----
def normal(mu, sigma, gen):
    u1, u2 = gen.rand(), gen.rand()
    z = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)
    return mu + sigma*z

# ---- Program ----
typ = input("1-Poisson, 2-Normalny: ")
n = int(input("Ile liczb: "))
seed = int(input("Ziarno: "))
gen = LCG(seed)

if typ == "1":
    l = float(input("Lambda: "))
    dane = [poisson(l, gen) for _ in range(n)]
    plt.hist(dane, bins=range(min(dane), max(dane)+2))
else:
    mu = float(input("Mu: "))
    sigma = float(input("Sigma: "))
    dane = [normal(mu, sigma, gen) for _ in range(n)]
    plt.hist(dane, bins=30)

plt.show()
