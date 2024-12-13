import numpy as np

WINDOWSIZE = 3
PRE_INDEX = 3
Xdata = []
Ydata = []
data = [1,2,3,4,5,6,7,8,9,10]

i = 0
while i < len(data)-PRE_INDEX:
    Xdata.append(data[i:i + WINDOWSIZE])
    Ydata.append(data[i+PRE_INDEX])
    i += 1
print(Xdata)
print(Ydata)