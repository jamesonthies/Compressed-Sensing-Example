#Jameson Thies
#EEC 266
#Fall 2018

#includes
import random
import math
import dct_lib
import matplotlib
import matplotlib.pyplot as plt

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
print('Data Creation Complete.')

#computing dct
dct_data = dct_lib.dct(original_data)
print('DCT Complete.')

#displaying l1 norms of original data and DCT of the data
print('l1 norm of original data: ', sum([abs(x) for x in original_data]))
print('l1 norm of dct(data): ', sum([abs(x) for x in dct_data]))

#ploting original signal
plt.subplot(2,1,1)
plt.title("Original Data")
original_data_plot = plt.plot(range(len(original_data)), original_data)
plt.xlim(0, 1023)
plt.setp(original_data_plot, linewidth=0.9)

#plotting dct of original signal
plt.subplot(2,1,2)
plt.title("DCT of Original Data")
dct_data_plot = plt.stem(range(len(dct_data)), dct_data,  markerfmt='C0.', basefmt="C0-")
plt.setp(dct_data_plot, linewidth=1)
plt.ylim(-11.5, 19)
plt.xlim(0, 1023)
#plt.setp(dct_data_plot,)
plt.show()
