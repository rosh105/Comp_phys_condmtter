import numpy as np
import matplotlib.pyplot as plt
def tail_DOS(E):
    g_0 = 1 #Placeholder value
    E_c = 1 #Placeholder value
    sigma = 1 #Placeholder value
    return g_0*np.exp((E-E_c)/sigma)
def gauss_DOS(E):
    g_0 = 1 #Placeholder value
    E_c = 1 #Placeholder value
    sigma = 1 #Placeholder value
    return g_0*np.exp(-(E-E_c)**2/2*sigma**2)

print(tail_DOS(-0.1))
print(gauss_DOS(-0.1))