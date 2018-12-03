#Jameson Thies
#EEC 266
#Fall 2018

#includes
import random
import math
import dct_lib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

#signal parameters
M = 512
T = 0.064
Fs = 16000
a1 = 1
f1 = 310*(2*math.pi)
a2 = 0.6
f2 = 540*(2*math.pi)
noise = 0.05

#creating signal
seed = 1
random.seed(seed)
original_x = ([x/Fs for x in range(round(T*Fs))])
original_data = [a1*math.sin(f1*x)+a2*math.sin(f2*x)+random.gauss(0,noise) for x in original_x]
print('Signal Created.')

#computing dct
dct_data = dct_lib.dct(original_data)
print('DCT Computed.')

#multiplying sparse representation of data by CS matrix to compress  data
np.random.seed(seed)
random_gaussian_matrix = np.matrix([[random.gauss(0,1) for x in range(len(original_data))] for i in range(M)])
dct_data_transpose = np.matrix([[d] for d in dct_data])
y = random_gaussian_matrix*dct_data_transpose
print('Data Compressed.')

#Here sklearn's orthogonal matching pursuit function is used to find the most l1 sparse solution to y.
#For this program, M nonzeros elements are assumed to be in the sparse representation.
OMP = OrthogonalMatchingPursuit(n_nonzero_coefs=M)
OMP.fit(random_gaussian_matrix, y)
omp_out = OMP.coef_
result = dct_lib.idct(omp_out)
print('OMP Computed.')
print("CR: ", len(original_data)/M)

#plotting original signal
plt.subplot(4, 1, 1)
plt.title("Original Data")
original_data_plot = plt.plot(range(len(original_data)), original_data)
plt.xlim(0, 1023)
plt.setp(original_data_plot, linewidth=0.9)

#plotting dct of original signal
plt.subplot(4, 1, 2)
plt.title("DCT of Original Data")
dct_data_plot = plt.stem(range(len(dct_data)), dct_data,  markerfmt='C0.', basefmt="C0-")
plt.setp(dct_data_plot, linewidth=1)
plt.ylim(-11.5, 19)
plt.xlim(0, 1023)

#plotting results of orthogonal matching pursuit.
plt.subplot(4, 1, 3)
plt.title("Recovered DCT Coefficients")
omp_out_plot = plt.stem(range(len(omp_out)), omp_out,  markerfmt='C0.', basefmt="C0-")
plt.setp(omp_out_plot, linewidth=1)
plt.ylim(-12.5, 22)
plt.xlim(0, 1023)

#plotting reconstructed signal
plt.subplot(4, 1, 4)
plt.title("Recovered Signal")
original_data_plot = plt.plot(range(len(original_data)), original_data)
result_data_plot = plt.plot(range(len(original_data)), result) 
plt.setp(original_data_plot, linewidth=0.9)
plt.setp(result_data_plot, linewidth=0.9)
plt.xlim(0, 1023)

plt.show()