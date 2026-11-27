from PIL import Image
import numpy as np

im = Image.open(r'c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\catan_mountain_normal_4k.png')
print('Normal map size:', im.size)
arr = np.array(im)
print('Normal min:', arr.min(axis=(0,1)), 'max:', arr.max(axis=(0,1)))

# Let's crop a 500x500 patch and check variations
crop = arr[1500:2000, 1500:2000]
im_crop = Image.fromarray(crop)
im_crop.save('scratch/normal_crop.png')
print('Saved scratch/normal_crop.png')
