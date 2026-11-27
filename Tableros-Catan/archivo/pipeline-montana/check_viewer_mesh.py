with open(r'c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\catan_mountain_viewer.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if any(k in l for k in ['PlaneGeometry', 'BufferGeometry', 'displacementMap', 'vertexShader', 'DEM_GRID', 'terrain', 'MeshStandardMaterial', 'texture']):
        if len(l) < 200:
            print(f'{i+1}: {l.strip()}')
        else:
            print(f'{i+1}: {l[:150]}...')
