import numpy as np
import time
import os

t0 = time.time()
num_triangles = 2283450
record_dtype = np.dtype([
    ('normal', ('<f4', 3)),
    ('v1', ('<f4', 3)),
    ('v2', ('<f4', 3)),
    ('v3', ('<f4', 3)),
    ('attr', '<u2')
])
stl_data = np.zeros(num_triangles, dtype=record_dtype)
header = b'Binary STL 2.28M Triangles' + b'\x00' * 54
count = np.uint32(num_triangles).tobytes()

fname = 'scratch/test_speed.stl'
with open(fname, 'wb') as f:
    f.write(header)
    f.write(count)
    stl_data.tofile(f)

sz = os.path.getsize(fname) / (1024 * 1024)
print(f"Wrote {num_triangles} triangles in {time.time()-t0:.2f}s! Size: {sz:.2f} MB")
os.remove(fname)
