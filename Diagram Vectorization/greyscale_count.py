import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_grayscale_pixel_counts(gray: np.ndarray, title="Grayscale pixel counts"):
    """
    gray: (H, W) grayscale image, ideally uint8
    Plots: x = intensity 0..255, y = number of pixels with that intensity
    """
    if gray.ndim != 2:
        raise ValueError("Expected a single-channel grayscale image (H, W).")

    # Ensure uint8 range 0..255
    if gray.dtype != np.uint8:
        g = np.clip(gray.astype(np.float64), 0, 255).astype(np.uint8)
    else:
        g = gray

    counts = np.bincount(g.ravel(), minlength=256)  # counts[ intensity ]
    intensities = np.arange(256)

    plt.figure()
    plt.bar(intensities, counts, width=1.0)
    plt.title(title)
    plt.xlabel("Grayscale intensity (0–255)")
    plt.ylabel("Number of pixels")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()

    return counts

# Example usage:
gray = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
counts = plot_grayscale_pixel_counts(gray, title="Histogram: input.png")
