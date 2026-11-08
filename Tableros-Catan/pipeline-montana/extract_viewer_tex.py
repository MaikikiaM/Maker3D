import base64
from PIL import Image
import numpy as np

with open(r'c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\catan_mountain_viewer.html', 'r', encoding='utf-8') as f:
    html = f.read()

prefix = 'const DEM_TEX_B64 = "data:image/png;base64,'
idx = html.find(prefix)
if idx != -1:
    end = html.find('"', idx + len(prefix))
    b64_str = html[idx+len(prefix):end]
    data = base64.b64decode(b64_str)
    with open('scratch/viewer_embedded_tex.png', 'wb') as f_out:
        f_out.write(data)
    im = Image.open('scratch/viewer_embedded_tex.png')
    arr = np.array(im)
    print('Saved scratch/viewer_embedded_tex.png', im.size, im.mode)
    green_mask = (arr[:, :, 1] > arr[:, :, 0] + 20) & (arr[:, :, 1] > arr[:, :, 2] + 20)
    print('Green pixels:', np.sum(green_mask))
    if np.sum(green_mask) > 0:
        ys, xs = np.where(green_mask)
        print('Y range:', ys.min(), ys.max())
        print('X range:', xs.min(), xs.max())
        print('Sample green RGB:', arr[ys[0], xs[0]])
