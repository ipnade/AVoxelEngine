import numpy as np
import glm
from voxel_chunk import Chunk

class World:
    GRID_SIZE = 4  # 4x4 grid

    def __init__(self):
        self.chunks = {}
        for gx in range(self.GRID_SIZE):
            for gz in range(self.GRID_SIZE):
                # Each chunk is offset by its grid position.
                offset = (gx * Chunk.WIDTH, 0, gz * Chunk.DEPTH)
                self.chunks[(gx, gz)] = Chunk(offset)

    def generate_mesh(self):
        """Combine mesh data from all chunks into one numpy array."""
        combined = []
        for chunk in self.chunks.values():
            combined.extend(chunk.generate_mesh())
        return np.array(combined, dtype="f4")

    def get_chunk_at(self, world_pos):
        """Return the chunk containing 'world_pos' (a tuple of ints)."""
        for chunk in self.chunks.values():
            ox, oy, oz = chunk.offset
            if (ox <= world_pos[0] < ox + Chunk.WIDTH and
                oy <= world_pos[1] < oy + Chunk.HEIGHT and
                oz <= world_pos[2] < oz + Chunk.DEPTH):
                return chunk
        return None

    def ray_voxel_traversal(self, ray_origin, ray_direction, max_distance=128):
        """
        Casts a ray through each chunk (by transforming ray_origin into local coordinates)
        and returns the closest hit voxel in world coordinates.
        """
        closest_world_hit = None
        closest_normal = None
        closest_dist = max_distance

        from raycast import ray_voxel_traversal  # reuse existing function

        for chunk in self.chunks.values():
            # Transform ray origin into local chunk coordinates.
            local_origin = glm.vec3(
                ray_origin.x - chunk.offset[0],
                ray_origin.y - chunk.offset[1],
                ray_origin.z - chunk.offset[2]
            )
            hit, normal = ray_voxel_traversal(local_origin, ray_direction, chunk)
            if hit is not None:
                # Convert hit position back to world coordinates.
                world_hit = glm.vec3(
                    hit[0] + chunk.offset[0],
                    hit[1] + chunk.offset[1],
                    hit[2] + chunk.offset[2]
                )
                dist = glm.distance(ray_origin, world_hit)
                if dist < closest_dist:
                    closest_world_hit = (int(world_hit.x), int(world_hit.y), int(world_hit.z))
                    closest_normal = normal
                    closest_dist = dist
        return closest_world_hit, closest_normal

    def any_chunk_needs_update(self):
        for chunk in self.chunks.values():
            if chunk.needs_update:
                return True
        return False

    def clear_update_flags(self):
        for chunk in self.chunks.values():
            chunk.needs_update = False