from turtle import width
import cynetworkx as netx
import numpy as np
import pickle 
import cv2
from queue import Queue

canvas_size = 256
block_size = 16


def out_of_bounds(x, y, cx, cy):
    return (abs(x - cx) >= canvas_size // 2) or (abs(y - cy) >= canvas_size // 2)


def generate_mesh(depth, canvas_min_x, canvas_min_y, real_min_x, real_min_y, blocks_x, blocks_y):
    verts = []
    idx = []
    verts1 = np.zeros((blocks_x, blocks_y, 5), dtype=np.float32)
    verts2 = np.zeros((blocks_x, blocks_y, 5), dtype=np.float32)
    for i in range(blocks_x):
        for j in range(blocks_y):
            verts1[i][j] = (real_min_x + (i * block_size), 
                            real_min_y + (j * block_size), 
                            depth[canvas_min_x + (i * block_size)][canvas_min_y + (j * block_size)], 
                            i / blocks_x,
                            j / blocks_y)

    for i in range(blocks_x):
        for j in range(blocks_y):
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
                        verts2[i][j][2] = verts1[i - 1][j - 1][2]
                        continue
                if i > 0 and j < blocks_y - 1:
                    if verts1[i - 1][j + 1][2] >= 0:
                        verts2[i][j][2] = verts1[i - 1][j + 1][2]
                        continue
                if j < blocks_y - 1:
                    if verts1[i][j + 1][2] >= 0:
                        verts2[i][j][2] = verts1[i][j + 1][2]
                        continue
                if i < blocks_x -1 and j > 0:
                    if verts1[i + 1][j - 1][2] >= 0:
                        verts2[i][j][2] = verts1[i + 1][j - 1][2]
                        continue
                if i < blocks_x -1:
                    if verts1[i + 1][j][2] >= 0:
                        verts2[i][j][2] = verts1[i + 1][j][2]
                        continue
                if i < blocks_x -1 and j < blocks_y - 1:
                    if verts1[i + 1][j + 1][2] >= 0:
                        verts2[i][j][2] = verts1[i + 1][j + 1][2]
                        continue

    k_00, k_02, k_11, k_12 = \
        g.graph['cam_param_pix_inv'][0, 0], g.graph['cam_param_pix_inv'][0, 2], \
        g.graph['cam_param_pix_inv'][1, 1], g.graph['cam_param_pix_inv'][1, 2]

    for i in range(blocks_x):
        for j in range(blocks_y):
            n = verts2[i][j]
            n[0] = n[2] * ((n[0] - g.graph['woffset']) * k_00 + k_02)
            n[1] = n[2] * ((n[1] - g.graph['hoffset']) * k_11 + k_12)
            verts.append(n)
            
    for i in range(blocks_x - 1):
        for j in range(blocks_y - 1):
            idx.append((i + j * blocks_x + 1, (i + 1) + j * blocks_x + 1, i + (j + 1) * blocks_x + 1))
            idx.append(((i + 1) + j * blocks_x + 1, (i + 1) + (j + 1) * blocks_x + 1, i + (j + 1) * blocks_x + 1))
    return verts, idx
    

def get_glyphs(g):
    glyphs = []
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    depth = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    depth -= 1
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
            depth[nx][ny] = abs(nd)            
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

            verts, idx = generate_mesh(depth, canvas_min_x, canvas_min_y, real_min_x, real_min_y, blocks_x, blocks_y)
            img = canvas[canvas_min_x : canvas_max_x, canvas_min_y : canvas_max_y]

            glyph = {'min_x': real_min_x,
                    'min_y': real_min_y,
                    'width': blocks_x,
                    'height': blocks_y,
                    'tex': img,
                    'verts': verts,
                    'idx': idx}
            glyphs.append(glyph)
            #cv2.imwrite(f"test/{i}.png", img)
            i+=1
            canvas[::] = 0
            depth[::] = -1

    glyphs.sort(key=lambda d: d['height'])
    return glyphs



f = open('mesh_small.gz', 'rb')
g = pickle.load(f)
f.close()

print(get_glyphs(g))
