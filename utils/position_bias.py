import numpy as np
'''
position bias is 1D np.array
'''

def singular(len):
    positionBias = np.zeros(len)
    positionBias[0] = 1
    return positionBias

def geometric(len,k,p):
    if(len<=0):
        print("parameter len shouldn't be less equal than 0. We reset it to 5.");
        len = 5
    if(k>len):
        print("parameter k shouldn't be greater than len. We reset it to len.")
        k = len
    if(p<=0 or p>=1):
        print("parameter p should between 0 and 1. We reset it to 0.5.")
        p = 0.5
    positionBias = np.zeros(len)
    positionBias[0] = p
    for j in range(1,k):
        positionBias[j] = positionBias[j-1]*(1-p)
    return positionBias
    
def log(cutoff):
    return (1/np.log2(2+np.arange(cutoff)))