import numpy as np
data = np.load('feature_out/road_accident_test.npy')
print(type(data))
print(len(data))

data = np.squeeze(data)
print(type(data))
print(len(data))

np.savetxt('feature_out/road_accident_test.txt',data, delimiter=' ')
