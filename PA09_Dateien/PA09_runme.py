# Python program to read and perform Fourier Transform on an image using Matplotlib and Numpy

# Importing necessary modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nf
import os



# Definition FFT (1D Fast Fourier Transform)

# Write your code here
"""
def fft(x):
    n = len(x)
    if n == 1:
        return x
    omega = exp(2*j * pi / n)
    x_even = FFT(x[0::2])
    x_odd = FFT(x[1::2])
    y_e = FFT(x_even)
    y_o = FFT(x_odd)
    y = [0] * n
    for k in range(n // 2):
        y[k] = y_e[k] + omega ** k * y_o[k]
        y[k + n // 2] = y_e[k] - omega ** k * y_o[k]
    return y

def fft(x):
    n = len(x)
    if n <= 1:
        return x

    i = 2
    while i < n:
        i *= 2
    n = i

    M = n//2
    omega = np.exp(-2j * np.pi / n)
    z = []
    for l in range(M):
        z.append(x[l] + x[l + M])
    c_even = fft(z)
    z = []
    for l in range(M):
        z.append((x[l] - x[l + M])*omega**l)
    c_odd = fft(z)
    c = []
    for l in range(M):
        c.append(n*c_even[l])
        c.append(n*c_odd[l])
    return c
    """

def fft(x):
    n = len(x)
    if n <= 1:
        return x
    
    # Padding auf Zweierpotenz
    if not (n & (n - 1)) == 0:  # Prüft, ob n keine Zweierpotenz ist
        next_power_of_2 = 1 << (n - 1).bit_length()
        x = np.pad(x, (0, next_power_of_2 - n), 'constant')
        n = next_power_of_2

    M = n // 2
    omega_n = np.exp(-2j * np.pi / n)

    # Geraden Indizes
    z_even = [x[l] + x[l + M] for l in range(M)]
    c_even = fft(z_even)

    # Ungeraden Indizes
    z_odd = [(x[l] - x[l + M]) * omega_n**l for l in range(M)]
    c_odd = fft(z_odd)

    # Kombinieren der Ergebnisse
    result = [0] * n
    for l in range(M):
        result[l] = c_even[l] + c_odd[l]
        result[l + M] = c_even[l] - c_odd[l]

    return result



# Definition of FFT2 (2D Fast Fourier Transform)

 # Write your code here
def fft2(matrix):
    # Transformiere jede Reihe
    matrix = np.array([fft(row) for row in matrix])
    # Transformiere jede Spalte
    matrix = np.array([fft(column) for column in matrix.T]).T
    return matrix

# Der Error in meinem  System war: "This probably means that Tcl wasn't installed properly."
# Ich konnte die  Ursache nicht finden. Wodurch ich meinen Code nicht debuggen konnte. 

###########################################################################################################
 
# RUNME


# Read the image
# Load the original image B1 from the same folder as the script
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'PA09_waves.jpg')
imageB1_int = mpimg.imread(image_path)
#imageB1_int = mpimg.imread('PA11_waves.jpg')

# Save image as a matrix
imageB1 = np.float64(imageB1_int)
print(imageB1)
# Perform a 2D Fast Fourier Transform (FFT) on the image
# fourier = nf.fft2(imageB1)
fourier = fft2(imageB1)
print(fourier)
# Separate Fourier transform into amplitude and phase
amplitude_B1 = np.absolute(fourier)
phase_B1 = np.angle(fourier)

# Show the original image
plt.figure(figsize=(15,10))
plt.subplot(1, 3, 1)
plt.title('Original Image B1')
plt.imshow(imageB1, cmap='gray', vmin=0, vmax=256)

# Display the amplitude of the Fourier Transform
cmin = np.min(amplitude_B1)
cmax = np.max(amplitude_B1)
plt.subplot(1, 3, 2)
plt.title('Amplitude of Fourier Transform F(B1)')
plt.imshow(np.absolute(nf.fftshift(fourier)), cmap='gray', vmin=cmin, vmax=cmax)

# Display the phase of the Fourier Transform
dmin = np.min(phase_B1)
dmax = np.max(phase_B1)
plt.subplot(1, 3, 3)
plt.title('Phase of Fourier Transform F(B1)')
plt.imshow(np.angle(nf.fftshift(fourier)), cmap='gray', vmin=dmin, vmax=dmax)
plt.show()

# Filtering the spectrum to keep only the largest and second-largest amplitudes
M1 = np.max(amplitude_B1)
M2 = np.max(amplitude_B1[amplitude_B1 < M1])
M3 = np.max(amplitude_B1[amplitude_B1 < M2])

amplitude_B2 = amplitude_B1
amplitude_B2[amplitude_B2 < (1 - 1e-15) * M2] = 0  # Set all small amplitudes to 0

# Transform the filtered Fourier transform back to the spatial domain to create image B2
imageB2 = nf.ifft2(np.multiply(amplitude_B2, np.exp(phase_B1 * 1j)))

# Discard noise in the imaginary part
if np.max(imageB2.imag) < 1e-12:
    imageB2 = imageB2.real

# Display B2 and its Fourier Transform
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(imageB2, cmap='gray', vmin=0, vmax=256)
plt.title('Image B2 (Filtered from B1)')

plt.subplot(1, 3, 2)
plt.imshow(nf.fftshift(amplitude_B2), cmap='gray', vmin=cmin, vmax=cmax)
plt.title('Amplitude of Fourier Transform F(B2)')

plt.subplot(1, 3, 3)
plt.imshow(nf.fftshift(phase_B1), cmap='gray', vmin=dmin, vmax=dmax)
# F(B1) and F(B2) have the same phase
plt.title('Phase of Fourier Transform F(B2)')
plt.show()

# Further filtering based on the third largest amplitude
amplitude_B3 = np.absolute(fourier)
amplitude_B3[amplitude_B3 < (1 - 1e-15) * M3] = 0  # Set all small amplitudes to 0
amplitude_B3[(amplitude_B3 > (1 + 1e-15) * M3) & (amplitude_B3 != M1)] = 0

imageB3 = nf.ifft2(np.multiply(amplitude_B3, np.exp(phase_B1 * 1j)))

# Discard noise in the imaginary part
if np.max(imageB3.imag) < 1e-12:
    imageB3 = imageB3.real

# Display B3 and its Fourier Transform
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(imageB3, cmap='gray', vmin=0, vmax=256)
plt.title('Image B3 (Filtered from B1)')

plt.subplot(1, 3, 2)
plt.imshow(nf.fftshift(amplitude_B3), cmap='gray', vmin=cmin, vmax=cmax)
plt.title('Amplitude of Fourier Transform F(B3)')

plt.subplot(1, 3, 3)
plt.imshow(nf.fftshift(phase_B1), cmap='gray', vmin=dmin, vmax=dmax)
# F(B1) and F(B3) have the same phase
plt.title('Phase of Fourier Transform F(B3)')
plt.show()


