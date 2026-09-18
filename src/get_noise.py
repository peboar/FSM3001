import numpy as np
import matplotlib.pyplot as plt

# 1. Load and prepare baseline image
img_path = r"/home/per/Desktop/Kth/Phd/Courses/FSM3001/Project/src/CT.png"
raw_img = plt.imread(img_path)
if raw_img.ndim == 3:
    raw_img = np.mean(raw_img, axis=2)

# Ensure data is strictly scaled 0.0 to 1.0 float for blending math
img_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min())

# 2. Compute 2D FFT and isolate the high-frequency map
f_output = np.fft.fftshift(np.fft.fft2(img_norm))
rows, cols = img_norm.shape
crow, ccol = rows // 2, cols // 2

# Create a high-pass frequency filter mask (10% of image width)
r_cutoff = int(cols * 0.10)
y_idx, x_idx = np.ogrid[:rows, :cols]
hp_filter = (x_idx - ccol)**2 + (y_idx - crow)**2 > r_cutoff**2

# Reconstruct the pure structural texture/roughness distribution map
f_filtered = f_output * hp_filter
roughness_map = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

# Normalize the roughness map to a 0-1 scale
roughness_norm = (roughness_map - roughness_map.min()) / (roughness_map.max() - roughness_map.min())

# 3. Apply the FFT distribution to an RGB color space
color_output = np.zeros((rows, cols, 3))

# Rule: High-frequency texture maps to Red channel (structural transitions)
color_output[:, :, 0] = roughness_norm * 0.95
# Rule: Original density maps to Green and Blue (gives a cool cyan binder matrix)
color_output[:, :, 1] = img_norm * 0.70
color_output[:, :, 2] = img_norm * 0.85

# 4. Display the structural frequency map
plt.figure(figsize=(8, 8))
plt.imshow(np.clip(color_output, 0, 1))
plt.title("FFT-Driven Texture Color Map")
plt.axis('off')
plt.show()
