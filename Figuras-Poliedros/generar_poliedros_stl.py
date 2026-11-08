import math
import struct
import os

def norm(v):
    l = math.sqrt(sum(x * x for x in v))
    if l == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)

def cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    )

def sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])

def add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])

def scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)

def dot(u, v):
    return sum(a * b for a, b in zip(u, v))

def make_triangle_outward(v1, v2, v3, centroid):
    n = cross(sub(v2, v1), sub(v3, v1))
    mid = scale(add(add(v1, v2), v3), 1.0 / 3.0)
    to_mid = sub(mid, centroid)
    if dot(n, to_mid) < 0:
        v2, v3 = v3, v2
        n = cross(sub(v2, v1), sub(v3, v1))
    return norm(n), v1, v2, v3

def write_stl(filename, triangles, description="Model"):
    header_bytes = f"{description}".encode('ascii')[:80]
    header = header_bytes + b'\0' * (80 - len(header_bytes))
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', len(triangles)))
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))
    print(f"Generated {filename}: {len(triangles)} triangles")

# -------------------------------------------------------------
# 1. CUBO (HEXAEDRO)
# -------------------------------------------------------------
def generate_cube(side=40.0):
    half = side / 2.0
    # Centered at X=0, Y=0, Z from 0 to side
    v = [
        (-half, -half, 0.0),   # 0
        ( half, -half, 0.0),   # 1
        ( half,  half, 0.0),   # 2
        (-half,  half, 0.0),   # 3
        (-half, -half, side),  # 4
        ( half, -half, side),  # 5
        ( half,  half, side),  # 6
        (-half,  half, side),  # 7
    ]
    centroid = (0.0, 0.0, side / 2.0)
    
    quads = [
        (0, 3, 2, 1), # bottom (-Z)
        (4, 5, 6, 7), # top (+Z)
        (0, 1, 5, 4), # front (-Y)
        (2, 3, 7, 6), # back (+Y)
        (0, 4, 7, 3), # left (-X)
        (1, 2, 6, 5), # right (+X)
    ]
    triangles = []
    for q in quads:
        triangles.append(make_triangle_outward(v[q[0]], v[q[1]], v[q[2]], centroid))
        triangles.append(make_triangle_outward(v[q[0]], v[q[2]], v[q[3]], centroid))
    return triangles

# -------------------------------------------------------------
# 2. TETRAEDRO REGULAR
# -------------------------------------------------------------
def generate_tetrahedron(edge=50.0):
    # Base is an equilateral triangle on Z=0, centered at (0, 0)
    # Circumradius of base: R = edge / sqrt(3)
    r = edge / math.sqrt(3.0)
    h = edge * math.sqrt(2.0 / 3.0) # approx 40.82 mm for edge 50 mm
    
    v0 = (0.0, r, 0.0)
    v1 = (-edge / 2.0, -r / 2.0, 0.0)
    v2 = ( edge / 2.0, -r / 2.0, 0.0)
    v3 = (0.0, 0.0, h) # Apex
    
    centroid = (0.0, 0.0, h / 4.0)
    faces = [
        (v0, v2, v1), # base
        (v0, v1, v3), # side 1
        (v1, v2, v3), # side 2
        (v2, v0, v3), # side 3
    ]
    triangles = [make_triangle_outward(f[0], f[1], f[2], centroid) for f in faces]
    return triangles

# -------------------------------------------------------------
# 3. OCTAEDRO REGULAR (Apoyado sobre una cara para impresión FDM)
# -------------------------------------------------------------
def generate_octahedron(edge=40.0):
    # Regular octahedron resting on a face is an antiprism
    r = edge / math.sqrt(3.0)
    h = edge * math.sqrt(2.0 / 3.0) # Height between opposite parallel faces
    
    # Bottom face at Z=0
    v0 = (r * math.cos(0.0), r * math.sin(0.0), 0.0)
    v1 = (r * math.cos(2.0 * math.pi / 3.0), r * math.sin(2.0 * math.pi / 3.0), 0.0)
    v2 = (r * math.cos(4.0 * math.pi / 3.0), r * math.sin(4.0 * math.pi / 3.0), 0.0)
    
    # Top face at Z=h (rotated 60 deg)
    u0 = (r * math.cos(math.pi / 3.0), r * math.sin(math.pi / 3.0), h)
    u1 = (r * math.cos(math.pi), r * math.sin(math.pi), h)
    u2 = (r * math.cos(5.0 * math.pi / 3.0), r * math.sin(5.0 * math.pi / 3.0), h)
    
    centroid = (0.0, 0.0, h / 2.0)
    faces = [
        (v0, v2, v1), # bottom base
        (u0, u1, u2), # top base
        (v0, v1, u0), # lateral 1
        (v1, v2, u1), # lateral 2
        (v2, v0, u2), # lateral 3
        (u0, u1, v1), # lateral 4
        (u1, u2, v2), # lateral 5
        (u2, u0, v0), # lateral 6
    ]
    triangles = [make_triangle_outward(f[0], f[1], f[2], centroid) for f in faces]
    return triangles

def generate_octahedron_vertex(edge=40.0):
    r = edge / math.sqrt(2.0)
    H = edge * math.sqrt(2.0)
    v_bot = (0.0, 0.0, 0.0)
    v_top = (0.0, 0.0, H)
    v_mid = [
        ( r, 0.0, r),
        (0.0,  r, r),
        (-r, 0.0, r),
        (0.0, -r, r),
    ]
    centroid = (0.0, 0.0, r)
    faces = [
        (v_bot, v_mid[1], v_mid[0]),
        (v_bot, v_mid[2], v_mid[1]),
        (v_bot, v_mid[3], v_mid[2]),
        (v_bot, v_mid[0], v_mid[3]),
        (v_top, v_mid[0], v_mid[1]),
        (v_top, v_mid[1], v_mid[2]),
        (v_top, v_mid[2], v_mid[3]),
        (v_top, v_mid[3], v_mid[0]),
    ]
    triangles = [make_triangle_outward(f[0], f[1], f[2], centroid) for f in faces]
    return triangles

# -------------------------------------------------------------
# 4. CONO RECTO
# -------------------------------------------------------------
def generate_cone(diameter=40.0, height=50.0, segments=72):
    r = diameter / 2.0
    apex = (0.0, 0.0, height)
    center_base = (0.0, 0.0, 0.0)
    
    circle_pts = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        circle_pts.append((r * math.cos(theta), r * math.sin(theta), 0.0))
        
    triangles = []
    for i in range(segments):
        p1 = circle_pts[i]
        p2 = circle_pts[(i + 1) % segments]
        
        # Base triangle (facing down -Z)
        triangles.append(((0.0, 0.0, -1.0), center_base, p2, p1))
        
        # Lateral triangle (side)
        n = norm(cross(sub(p2, p1), sub(apex, p1)))
        triangles.append((n, p1, p2, apex))
        
    return triangles

# -------------------------------------------------------------
# 5. DODECAEDRO REGULAR (Apoyado sobre una cara pentagonal en Z=0)
# -------------------------------------------------------------
def generate_dodecahedron(edge=18.0):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    
    raw_verts = []
    for s1 in [-1.0, 1.0]:
        for s2 in [-1.0, 1.0]:
            for s3 in [-1.0, 1.0]:
                raw_verts.append((s1, s2, s3))
    for s1 in [-1.0, 1.0]:
        for s2 in [-1.0, 1.0]:
            raw_verts.append((0.0, s1 / phi, s2 * phi))
            raw_verts.append((s1 / phi, s2 * phi, 0.0))
            raw_verts.append((s1 * phi, 0.0, s2 / phi))
            
    raw_edge_len = 2.0 / phi
    scale_factor = edge / raw_edge_len
    verts = [scale(v, scale_factor) for v in raw_verts]
    
    # Adjacency
    adj = {i: [] for i in range(20)}
    for i in range(20):
        for j in range(i + 1, 20):
            if abs(math.dist(verts[i], verts[j]) - edge) < 1e-3:
                adj[i].append(j)
                adj[j].append(i)
                
    # Find 12 pentagonal cycles
    faces = []
    for i in range(20):
        for j in adj[i]:
            for k in adj[j]:
                if k == i: continue
                for l in adj[k]:
                    if l in (i, j): continue
                    for m in adj[l]:
                        if m in (i, j, k): continue
                        if i in adj[m]:
                            cycle_set = frozenset([i, j, k, l, m])
                            if not any(f['set'] == cycle_set for f in faces):
                                # Order vertices in cycle
                                faces.append({
                                    'set': cycle_set,
                                    'cycle': [i, j, k, l, m]
                                })
                                
    # Pick face 0 to be the base resting on Z=0
    base_face = faces[0]['cycle']
    base_center = scale(sum([verts[idx] for idx in base_face], (0.0, 0.0, 0.0)), 0.2)
    # Actually sum tuples:
    bc = [0.0, 0.0, 0.0]
    for idx in base_face:
        bc[0] += verts[idx][0] / 5.0
        bc[1] += verts[idx][1] / 5.0
        bc[2] += verts[idx][2] / 5.0
    base_center = tuple(bc)
    
    base_normal = norm(base_center) # from origin (0,0,0) to face center
    
    # We want to rotate so base_normal aligns with (0, 0, -1)
    target_normal = (0.0, 0.0, -1.0)
    rot_axis = cross(base_normal, target_normal)
    axis_len = math.sqrt(sum(x * x for x in rot_axis))
    c = dot(base_normal, target_normal)
    
    if axis_len > 1e-6:
        k = norm(rot_axis)
        s = axis_len
        def rotate(p):
            axp = cross(k, p)
            kdp = dot(k, p)
            return (
                p[0] * c + axp[0] * s + k[0] * kdp * (1.0 - c),
                p[1] * c + axp[1] * s + k[1] * kdp * (1.0 - c),
                p[2] * c + axp[2] * s + k[2] * kdp * (1.0 - c)
            )
    else:
        rotate = lambda p: p
        
    rotated_verts = [rotate(v) for v in verts]
    min_z = min(v[2] for v in rotated_verts)
    
    # Shift up so base is at Z=0
    final_verts = [(v[0], v[1], v[2] - min_z) for v in rotated_verts]
    centroid = (0.0, 0.0, -min_z)
    
    triangles = []
    for f in faces:
        c_idx = f['cycle']
        v_pts = [final_verts[idx] for idx in c_idx]
        # Triangulate pentagon into 3 triangles: (0, 1, 2), (0, 2, 3), (0, 3, 4)
        for tri in [(v_pts[0], v_pts[1], v_pts[2]),
                    (v_pts[0], v_pts[2], v_pts[3]),
                    (v_pts[0], v_pts[3], v_pts[4])]:
            triangles.append(make_triangle_outward(tri[0], tri[1], tri[2], centroid))
            
    return triangles

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    cube_tris = generate_cube(side=40.0)
    write_stl(os.path.join(out_dir, "cubo.stl"), cube_tris, "Cubo 40x40x40mm")
    
    tetra_tris = generate_tetrahedron(edge=50.0)
    write_stl(os.path.join(out_dir, "tetraedro.stl"), tetra_tris, "Tetraedro Arista 50mm")
    
    octa_tris = generate_octahedron(edge=40.0)
    write_stl(os.path.join(out_dir, "octaedro.stl"), octa_tris, "Octaedro Arista 40mm (Apoyo cara)")
    
    octa_vert_tris = generate_octahedron_vertex(edge=40.0)
    write_stl(os.path.join(out_dir, "octaedro_vertice.stl"), octa_vert_tris, "Octaedro Arista 40mm (Apoyo vertice)")
    
    cone_tris = generate_cone(diameter=40.0, height=50.0, segments=72)
    write_stl(os.path.join(out_dir, "cono_recto.stl"), cone_tris, "Cono Recto D40 H50mm")
    
    dode_tris = generate_dodecahedron(edge=18.0)
    write_stl(os.path.join(out_dir, "dodecaedro.stl"), dode_tris, "Dodecaedro Arista 18mm")
    
    print("All STL files generated successfully!")
