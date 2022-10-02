from turtle import width
import cynetworkx as netx
import numpy as np
import pickle 
import cv2
from queue import Queue

atlas_size = 1024
canvas_size = 256
block_size = 16
gap_thickness = 2

def out_of_bounds(x, y, cx, cy):
    return (abs(x - cx) >= (canvas_size // 2) - gap_thickness) or (abs(y - cy) >= (canvas_size // 2) - gap_thickness)


def generate_mesh(depth, real_min_x, real_min_y, verts_x, verts_y):
    verts = []
    idx = []
    verts1 = np.zeros((verts_x, verts_y, 5), dtype=np.float32)
    verts2 = np.zeros((verts_x, verts_y, 5), dtype=np.float32)
    for i in range(verts_x):
        for j in range(verts_y):
            offset_x = i * block_size if i < verts_x - 1 else i * block_size - 1
            offset_y = j * block_size if j < verts_y - 1 else j * block_size - 1
            verts1[i][j] = (real_min_x + offset_x, 
                            real_min_y + offset_y, 
                            depth[offset_x][offset_y],
                            i / (verts_x - 1),
                            j / (verts_y - 1))

    for i in range(verts_x):
        for j in range(verts_y):
            verts2[i][j] = verts1[i][j]
            if verts1[i][j][2] < 0:
                if i > 0 and j > 0:
                    if verts1[i - 1][j - 1][2] >= 0:
                        verts2[i][j][2] = verts1[i - 1][j - 1][2]
                        continue
                if i > 0:
                    if verts1[i - 1][j][2] >= 0:
                        verts2[i][j][2] = verts1[i - 1][j][2]
                        continue
                if j > 0:
                    if verts1[i][j - 1][2] >= 0:
                        verts2[i][j][2] = verts1[i][j - 1][2]
                        continue
                if i > 0 and j < verts_y - 1:
                    if verts1[i - 1][j + 1][2] >= 0:
                        verts2[i][j][2] = verts1[i - 1][j + 1][2]
                        continue
                if j < verts_y - 1:
                    if verts1[i][j + 1][2] >= 0:
                        verts2[i][j][2] = verts1[i][j + 1][2]
                        continue
                if i < verts_x -1 and j > 0:
                    if verts1[i + 1][j - 1][2] >= 0:
                        verts2[i][j][2] = verts1[i + 1][j - 1][2]
                        continue
                if i < verts_x -1:
                    if verts1[i + 1][j][2] >= 0:
                        verts2[i][j][2] = verts1[i + 1][j][2]
                        continue
                if i < verts_x -1 and j < verts_y - 1:
                    if verts1[i + 1][j + 1][2] >= 0:
                        verts2[i][j][2] = verts1[i + 1][j + 1][2]
                        continue

    k_00, k_02, k_11, k_12 = \
        g.graph['cam_param_pix_inv'][0, 0], g.graph['cam_param_pix_inv'][0, 2], \
        g.graph['cam_param_pix_inv'][1, 1], g.graph['cam_param_pix_inv'][1, 2]

    for i in range(verts_x):
        for j in range(verts_y):
            n = verts2[i][j]
            n[0] = n[2] * ((n[0] - g.graph['woffset']) * k_00 + k_02)
            n[1] = n[2] * ((n[1] - g.graph['hoffset']) * k_11 + k_12)
            verts.append(n)
            
    for i in range(verts_x - 1):
        for j in range(verts_y - 1):
            if verts2[i][j][2] >= 0 and verts2[i+1][j][2] >= 0 and verts2[i][j+1][2] >= 0 and verts2[i+1][j+1][2] >= 0:
                idx.append([i + j * verts_x + 1, (i + 1) + j * verts_x + 1, i + (j + 1) * verts_x + 1])
                idx.append([(i + 1) + j * verts_x + 1, (i + 1) + (j + 1) * verts_x + 1, i + (j + 1) * verts_x + 1])
    return verts, idx
    

def get_glyphs(g):
    glyphs = []
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    depth_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    depth_canvas -= 1
    q = Queue()
    next_q = Queue()
    visited = set()

    n = list(g)[0]
    next_q.put(n)
    i = 0

    while not next_q.empty():
        modified = False
        canvas_min_x = canvas_min_y = canvas_size
        canvas_max_x = canvas_max_y = -1
        n = next_q.get()
        cx, cy, _ = n
        cc = g.nodes[n]['cc_id']
        q.put(n)
        while not q.empty():
            n = q.get()
            if n in visited:
                continue

            nx, ny, nd = n
            if out_of_bounds(nx, ny, cx, cy):
                next_q.put(n)
                continue

            if cc != g.nodes[n]['cc_id']:
                next_q.put(n)
                continue

            visited.add(n)
            nx = nx - cx + (canvas_size // 2)
            ny = ny - cy + (canvas_size // 2)
            if nx > canvas_max_x:
                canvas_max_x = nx
            if nx < canvas_min_x:
                canvas_min_x = nx
            if ny > canvas_max_y:
                canvas_max_y = ny
            if ny < canvas_min_y:
                canvas_min_y = ny
            canvas[nx][ny] = np.append(g.nodes[n]['color'][::-1], 255)
            depth_canvas[nx][ny] = abs(nd)            
            modified = True

            for neighbour in g[n]:
                if neighbour not in visited:    
                    nx2, ny2, _ = neighbour
                    if out_of_bounds(nx2, ny2, cx, cy):
                        next_q.put(neighbour)
                    else:
                        q.put(neighbour)

            if 'far' in g.nodes[n] and g.nodes[n]['far'] is not None:
                canvas[nx][ny][3] = 128
                for neighbour in g.nodes[n]['far']:
                    next_q.put(neighbour)

            if 'near' in g.nodes[n] and g.nodes[n]['near'] is not None:
                for neighbour in g.nodes[n]['near']:
                    next_q.put(neighbour)

            
        if modified:
            canvas_min_x -= canvas_min_x % block_size
            canvas_min_y -= canvas_min_y % block_size
            canvas_max_x += 0 if canvas_max_x % block_size is 0 else block_size - (canvas_max_x % block_size)
            canvas_max_y += 0 if canvas_max_y % block_size is 0 else block_size - (canvas_max_y % block_size)
            real_min_x = canvas_min_x + cx - (canvas_size // 2)
            real_min_y = canvas_min_y + cy - (canvas_size // 2)
            blocks_x = (canvas_max_x - canvas_min_x) // block_size
            blocks_y = (canvas_max_y - canvas_min_y) // block_size

            img = canvas[canvas_min_x : canvas_max_x, canvas_min_y : canvas_max_y]            
            depth = depth_canvas[canvas_min_x : canvas_max_x, canvas_min_y : canvas_max_y]

            verts, idx = generate_mesh(depth, real_min_x, real_min_y, blocks_x + 1, blocks_y + 1)

            glyph = {"id": i,
                    'width': blocks_x,
                    'height': blocks_y,
                    'tex': np.copy(img),
                    'verts': verts,
                    'idx': idx}
            glyphs.append(glyph)
            #cv2.imwrite(f"test/{i}.png", img)
            #depth *= 64 
            #cv2.imwrite(f"test/{i}_d.png", depth)
            i+=1
            canvas[::] = 0
            depth_canvas[::] = -1

    glyphs.sort(key=lambda d: d['height'], reverse=True)
    return glyphs


def check_mask(m, i, j, width, height):
    clash = m[i][j] or m[i+width][j] or m[i][j+height] or m[i+width][j+height]
    return not clash


def place_glyph(glyph, atlas, mask, mask_size, verts, idx):
    w = glyph['width']
    h = glyph['height']
    for i in range(mask_size - w):
        for j in range(mask_size - h):
            if check_mask(mask, i, j, w, h):
                atlas[i * block_size : (i + w) * block_size, j * block_size : (j + h) * block_size, :] = glyph['tex']
                mask[i : i + w, j : j + h] = True
                for id in glyph['idx']:
                    id[0] += len(verts)
                    id[1] += len(verts)
                    id[2] += len(verts)
                    idx.append(id)
                for vert in glyph['verts']:
                    vert[3] = vert[3] * (w / mask_size) + (i / mask_size)
                    vert[4] = vert[4] * (h / mask_size) + (j / mask_size)
                    verts.append(vert)
                return
    raise "Not enough space in the atlas"


def create_atlas(g):
    glyphs = get_glyphs(g)
    mask_size = atlas_size // block_size
    atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
    mask = np.zeros((mask_size, mask_size), dtype=np.bool8)
    verts = []
    idx = []
    for glyph in glyphs:
        place_glyph(glyph, atlas, mask, mask_size, verts, idx)
    return atlas, verts, idx


f = open('mesh_small.gz', 'rb')
g = pickle.load(f)
f.close()
atlas, verts, idx = create_atlas(g)
cv2.imwrite(f"test/atlas.png", atlas)
