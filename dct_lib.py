#Jameson Thies
#EEC 266
#Fall 2018

#includes
import math

#computes dct of in_vec
def dct(in_vec):
	out_vec = []
	for i1 in range(len(in_vec)):
		temp = 0;
		for i2 in range(len(in_vec)):
			temp += in_vec[i2]*(1/math.sqrt(2) if (i1 == 1) else 1)*math.cos((math.pi/(2*len(in_vec)))*((2*i2)-1)*(i1-1))
		out_vec.append(temp*math.sqrt(2/len(in_vec)))
	return out_vec

#computes inverse dct of in_vec
def idct(in_vec):
	out_vec = []
	for i1 in range(len(in_vec)):
		temp = 0;
		for i2 in range(len(in_vec)):
			temp += in_vec[i2]*(1/math.sqrt(2) if (i2 == 1) else 1)*math.cos((math.pi/(2*len(in_vec)))*(i2-1)*((2*i1)-1))
		out_vec.append(temp*math.sqrt(2/len(in_vec)))
	return out_vec