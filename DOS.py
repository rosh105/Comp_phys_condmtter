import numpy as np
import matplotlib.pyplot as plt
E_0 = 1 #Conduction band energy
psi = np.zeros((100,100))#Placeholder for potential values
E_c = E_0 - psi
def tail_DOS(E):
    g_0 = 1 #Placeholder value
    sigma = 1 #Placeholder value
    return g_0*np.exp((E-E_c)/sigma)
def gauss_DOS(E):
    g_0 = 1 #Placeholder value
    sigma = 1 #Placeholder value
    return g_0*np.exp(-(E-E_c)**2/2*sigma**2)

