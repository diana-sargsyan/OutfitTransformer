import cv2
import numpy as np

def apply_bilateral_filtering(image, d=5, sigma_color=40, sigma_space=5):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_guided_filtering(image, radius=7, eps=1e-6):
    # Use the image itself as the guide image
    # For better results, you could create a separate guide image that emphasizes structure
    return cv2.guidedFilter(image, image, radius, eps)

def apply_frequency_filtering(image, cutoff_low=10, cutoff_high=80):
    # Convert to grayscale for simplicity
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # FFT transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Create a mask for band-pass filtering
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    
    # Create band-pass filter - this removes very low and very high frequencies
    mask[crow-cutoff_low:crow+cutoff_low, ccol-cutoff_low:ccol+cutoff_low] = 0  # Remove low frequencies
    mask[crow-cutoff_high:crow+cutoff_high, ccol-cutoff_high:ccol+cutoff_high] = 1  # Keep mid frequencies
    
    # Apply mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize and return
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)