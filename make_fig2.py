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

#calculates signal to noise and distortion ratio (SNDR)
def sndr(orig, recovered):
	norm1 = math.sqrt(sum([y**2 for y in orig]))
	norm2 = math.sqrt(sum([(orig[i]-recovered[i])**2 for i in range(len(orig))]))
	res = 20*math.log((norm1/norm2),10)
	return res

#signal parameters
M = 512
T = 0.064
Fs = 16000
a1 = 1
f1 = 310*(2*math.pi)
a2 = 0.6
f2 = 540*(2*math.pi)
noise = 0.05

#creating data
seed = 1
random.seed(seed)
original_x = ([x/Fs for x in range(round(T*Fs))])
original_data = [a1*math.sin(f1*x)+a2*math.sin(f2*x)+random.gauss(0,noise) for x in original_x]
print('Data Creation Complete.')

#computing dct of original data
dct_data = dct_lib.dct(original_data)
print('DCT Complete.')

#creating CS matrix
np.random.seed(seed)
random_gaussian_matrix = np.matrix([[random.gauss(0,1) for x in range(len(original_data))] for i in range(M)])
dct_data_transpose = np.matrix([[d] for d in dct_data])

#mutliplying the dct data by different sized portions of the CS matrix for multiple compression ratios.
#all portions of the CS matrix satisfy the properties of a CS matrix
y64 = random_gaussian_matrix[0:64,:]*dct_data_transpose
y128 = random_gaussian_matrix[0:128,:]*dct_data_transpose
y256 = random_gaussian_matrix[0:256,:]*dct_data_transpose
y512 = random_gaussian_matrix[0:512,:]*dct_data_transpose
print('Compressing Data Complete.')

#Here sklearn's orthogonal matching pursuit function is used to find the most l1 sparse solution to y.
#For this program, M nonzeros elements are assumed to be in the sparse representation.
#This is repeated to recover a signal from each of the four results of the CS matrix multiplication.
OMP64 = OrthogonalMatchingPursuit(n_nonzero_coefs=64)
OMP64.fit(random_gaussian_matrix[0:64,:], y64)
omp_out64 = OMP64.coef_
result64 = dct_lib.idct(omp_out64)

OMP128 = OrthogonalMatchingPursuit(n_nonzero_coefs=128)
OMP128.fit(random_gaussian_matrix[0:128,:], y128)
omp_out128 = OMP128.coef_
result128 = dct_lib.idct(omp_out128)

OMP256 = OrthogonalMatchingPursuit(n_nonzero_coefs=256)
OMP256.fit(random_gaussian_matrix[0:256,:], y256)
omp_out256 = OMP256.coef_
result256 = dct_lib.idct(omp_out256)

OMP512 = OrthogonalMatchingPursuit(n_nonzero_coefs=512)
OMP512.fit(random_gaussian_matrix[0:512,:], y512)
omp_out512 = OMP512.coef_
result512 = dct_lib.idct(omp_out512)
print('OMP Complete.')
print("CR: ", len(original_data)/M)
print('m = 64 SNDR: ', sndr(original_data, result64))
print('m = 128 SNDR: ', sndr(original_data, result128))
print('m = 256 SNDR: ', sndr(original_data, result256))
print('m = 512 SNDR: ', sndr(original_data, result512))

#plotting all 4 compression and recovery results.
lw = 0.8
plt.subplot(4, 1, 1)
plt.title("Recovered Signal. M = 64")
original_data_plot = plt.plot(range(len(original_x)), original_data)
result_data_plot64 = plt.plot(range(len(original_x)), result64) 
plt.xlim(0, 1023)
plt.setp(original_data_plot, linewidth=lw)
plt.setp(result_data_plot64, linewidth=lw)

plt.subplot(4, 1, 2)
plt.title("Recovered Signal. M = 128")
original_data_plot = plt.plot(range(len(original_x)), original_data)
result_data_plot128 = plt.plot(range(len(original_x)), result128) 
plt.xlim(0, 1023)
plt.setp(original_data_plot, linewidth=lw)
plt.setp(result_data_plot128, linewidth=lw)

plt.subplot(4, 1, 3)
plt.title("Recovered Signal. M = 256")
original_data_plot = plt.plot(range(len(original_x)), original_data)
result_data_plot256 = plt.plot(range(len(original_x)), result256) 
plt.xlim(0, 1023)
plt.setp(original_data_plot, linewidth=lw)
plt.setp(result_data_plot256, linewidth=lw)

plt.subplot(4, 1, 4)
plt.title("Recovered Signal. M = 512")
original_data_plot = plt.plot(range(len(original_x)), original_data)
result_data_plot512 = plt.plot(range(len(original_x)), result512) 
plt.xlim(0, 1023)
plt.setp(original_data_plot, linewidth=lw)
plt.setp(result_data_plot512, linewidth=lw)

plt.show()