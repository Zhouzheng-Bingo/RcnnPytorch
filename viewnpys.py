import numpy as np

npy_file_path = "_cache/pres.npy"

npy_data = np.load(npy_file_path)

print("Shape of the data:", npy_data.shape)
print("Data type:", npy_data.dtype)

print(" rows:")
print(npy_data[:5])
