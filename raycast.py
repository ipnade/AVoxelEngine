import numpy as np
import glm
from voxel_chunk import Chunk

# --- Helpers for ray-box intersection ---
def ray_box_intersection(ray_origin, ray_dir, box_min, box_max):
    # Convert inputs to numpy arrays for calculations
    origin = np.array([ray_origin.x, ray_origin.y, ray_origin.z], dtype='f4')
    direction = np.array([ray_dir.x, ray_dir.z, ray_dir.z], dtype='f4')
    box_min = np.array(box_min, dtype='f4')
    box_max = np.array(box_max, dtype='f4')

    tmin = (box_min - origin) / direction
    tmax = (box_max - origin) / direction
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    t_near = np.max(t1)
    t_far = np.min(t2)
    
    if t_far < 0 or t_near > t_far:
        return None
        
    # Determine hit face normal based on which axis t_near comes from
    epsilon = 1e-4
    normal = glm.vec3(0)  # Use GLM vector for normal
    for i in range(3):
        if abs((box_min[i] - origin[i]) - direction[i]*t_near) < epsilon:
            normal[i] = -1
            break
        if abs((box_max[i] - origin[i]) - direction[i]*t_near) < epsilon:
            normal[i] = 1
            break
    return t_near, normal

def ray_voxel_traversal(ray_origin, ray_direction, chunk, max_distance=128):
    # Convert inputs to glm vectors if they aren't already
    origin = glm.vec3(*ray_origin) if not isinstance(ray_origin, glm.vec3) else ray_origin
    direction = glm.normalize(glm.vec3(*ray_direction)) if not isinstance(ray_direction, glm.vec3) else ray_direction

    x, y, z = map(int, origin)
    step_x = 1 if direction.x >= 0 else -1
    step_y = 1 if direction.y >= 0 else -1
    step_z = 1 if direction.z >= 0 else -1

    def t_scale(i, o, d, step):
        boundary = i + (1 if step > 0 else 0)
        return (boundary - o) / (d + 1e-8)

    tMaxX = t_scale(x, origin.x, direction.x, step_x)
    tMaxY = t_scale(y, origin.y, direction.y, step_y)
    tMaxZ = t_scale(z, origin.z, direction.z, step_z)
    
    tDeltaX = abs(1.0 / (direction.x + 1e-8))
    tDeltaY = abs(1.0 / (direction.y + 1e-8))
    tDeltaZ = abs(1.0 / (direction.z + 1e-8))

    traveled = 0.0
    normal = glm.vec3(0)  # Instead of np.zeros(3)

    while traveled <= max_distance:
        if 0 <= x < Chunk.WIDTH and 0 <= y < Chunk.HEIGHT and 0 <= z < Chunk.DEPTH:
            if (x, y, z) in chunk.voxels:
                return (x, y, z), normal  # Return both hit position and normal

        if tMaxX < tMaxY and tMaxX < tMaxZ:
            traveled = tMaxX
            tMaxX += tDeltaX
            x += step_x
            normal = glm.vec3(-step_x, 0, 0)  # Create new vector instead of modifying
        elif tMaxY < tMaxZ:
            traveled = tMaxY
            tMaxY += tDeltaY
            y += step_y
            normal = glm.vec3(0, -step_y, 0)  # Create new vector instead of modifying
        else:
            traveled = tMaxZ
            tMaxZ += tDeltaZ
            z += step_z
            normal = glm.vec3(0, 0, -step_z)  # Create new vector instead of modifying

    return None, None  # Return None for both position and normal if no hit

# --- Helper: Return nearby voxels (including newly exposed ones) ---
def get_nearby_boundary_voxels(chunk, player_position, radius=5):
    near_voxels = []
    px, py, pz = map(int, player_position)
    for pos in chunk.get_voxel_positions():
        x, y, z = pos
        if abs(x - px) <= radius and abs(y - py) <= radius and abs(z - pz) <= radius:
            near_voxels.append(pos)
    return near_voxels