import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Load the data from the CSV file
file_path = 'c_15_50.csv'  # Replace with your file path
data = pd.read_csv(file_path,header=None)

# Extracting the second and last column (assuming Python's 0-indexing)
column_2 = data.iloc[:, 2]
column_minus_1 = data.iloc[:, 3]

# Compute the Fast Fourier Transform (FFT) for each column
fft_column_2 = fft(column_2)
fft_column_minus_1 = fft(column_minus_1)

# Compute the frequency bins
n = len(column_2)
freq = np.fft.fftfreq(n,1/400)

# Plotting the spectrum for each column
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(freq, np.abs(fft_column_2))
plt.title('Frequency Spectrum of 2nd Column')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.subplot(1, 2, 2)
plt.plot(freq, np.abs(fft_column_minus_1))
plt.title('Frequency Spectrum of -1st Column')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
