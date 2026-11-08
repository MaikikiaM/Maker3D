from PIL import Image
import numpy as np

im = Image.open('scratch/viewer_embedded_tex.png')
arr = np.array(im)
green_mask = (arr[:, :, 1] > arr[:, :, 0] + 15) & (arr[:, :, 1] > arr[:, :, 2] + 15)
ys, xs = np.where(green_mask)

# Find connected components or patches
print(f'Total green pixels: {len(ys)}')
# Let us crop around one of the green patches
y_mid = int(np.median(ys))
x_mid = int(np.median(xs))
patch = arr[max(0, y_mid-100):min(arr.shape[0], y_mid+100), max(0, x_mid-100):min(arr.shape[1], x_mid+100)]
Image.fromarray(patch).save('scratch/green_patch_crop.png')
print(f'Saved green patch crop around ({x_mid}, {y_mid})')
