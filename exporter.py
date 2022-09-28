import cynetworkx as netx
import numpy as np
import pickle 
import cv2
from queue import Queue

f = open('mesh_small.gz', 'rb')
g = pickle.load(f)
f.close()

canvas_size = 256
block_size = 16

canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
depth = np.zeros((canvas_size, canvas_size), dtype=np.float32)

def out_of_bounds(x, y, cx, cy):
    return (abs(x - cx) >= canvas_size // 2) or (abs(y - cy) >= canvas_size // 2)     

ccs = netx.connected_components(g)

i = 0
q = Queue()
next_q = Queue()
visited = set()

n = list(g)[0]
next_q.put(n)
i = 0

while not next_q.empty():
    modified = False
    min_x = min_y = canvas_size
    max_x = max_y = -1
    n = next_q.get()
    cx, cy, _ = n
    q.put(n)
    while not q.empty():
        n = q.get()
        if n in visited:
            continue

        nx, ny, nd = n
        if out_of_bounds(nx, ny, cx, cy):
            next_q.put(n)
            continue

        visited.add(n)
        nx = nx - cx + (canvas_size // 2)
        ny = ny - cy + (canvas_size // 2)
        if nx > max_x:
            max_x = nx
        if nx < min_x:
            min_x = nx
        if ny > max_y:
            max_y = ny
        if ny < min_y:
            min_y = ny
        canvas[nx][ny] = np.append(g.nodes[n]['color'][::-1], 255)
        depth[nx][ny] = nd            
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
        min_x -= min_x % block_size
        min_y -= min_y % block_size
        max_x += block_size - (max_x % block_size)
        max_y += block_size - (max_x % block_size)
        img = canvas[min_x : max_x, min_y : max_y]
        cv2.imwrite(f"test/{i}.png", img)
        i+=1
        canvas[::] = 0
        depth[::] = 0




'''
for cc in ccs:
    for (nx, ny, nd) in cc:
        key = (nx, ny, nd)
        if key not in visited:
'''
