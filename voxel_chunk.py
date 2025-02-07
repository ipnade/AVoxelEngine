import numpy as np

class Chunk:
    WIDTH = 16
    HEIGHT = 128
    DEPTH = 16

    def __init__(self):
        self.voxels = {}
        self._mesh_cache = None
        self._voxel_mesh_data = {}  # Stores mesh data for each voxel
        self.needs_update = False

        # Initialize chunk
        for x in range(self.WIDTH):
            for z in range(self.DEPTH):
                self.voxels[(x, 0, z)] = "Bedrock"
        for x in range(self.WIDTH):
            for y in range(1, 64):
                for z in range(self.DEPTH):
                    self.voxels[(x, y, z)] = "Stone"

        # Build initial meshes for all voxels
        self._build_all_voxel_meshes()
        self._update_full_mesh()
        self.needs_update = True

    def remove_voxel(self, pos):
        if pos in self.voxels:
            del self.voxels[pos]
            # Remove mesh data for this voxel
            self._voxel_mesh_data.pop(pos, None)
            # Update neighbors too, since they might now have visible faces
            self._update_voxel_and_neighbors(pos)
            self.needs_update = True

    def add_voxel(self, pos):
        if (0 <= pos[0] < self.WIDTH and 0 <= pos[1] < self.HEIGHT and 0 <= pos[2] < self.DEPTH):
            self.voxels[pos] = "Stone"
            # Rebuild mesh for this new voxel + neighbors
            self._update_voxel_and_neighbors(pos)
            self.needs_update = True

    def _build_all_voxel_meshes(self):
        """Generate meshes for all voxels."""
        self._voxel_mesh_data = {}
        for pos in self.voxels.keys():
            self._voxel_mesh_data[pos] = self._build_voxel_geometry(pos)

    def _build_voxel_geometry(self, pos):
        """Generate vertices+normals for the faces of one voxel at 'pos' that are visible."""
        x, y, z = pos
        faces = {
            "front":  ((0, 0, 1),  [(0,0,1),(1,0,1),(1,1,1),(0,0,1),(1,1,1),(0,1,1)]),
            "back":   ((0, 0, -1), [(1,0,0),(0,0,0),(0,1,0),(1,0,0),(0,1,0),(1,1,0)]),
            "left":   ((-1,0,0),   [(0,0,0),(0,0,1),(0,1,1),(0,0,0),(0,1,1),(0,1,0)]),
            "right":  ((1, 0, 0),  [(1,0,1),(1,0,0),(1,1,0),(1,0,1),(1,1,0),(1,1,1)]),
            "top":    ((0, 1, 0),  [(0,1,1),(1,1,1),(1,1,0),(0,1,1),(1,1,0),(0,1,0)]),
            "bottom": ((0, -1,0),  [(0,0,0),(1,0,0),(1,0,1),(0,0,0),(1,0,1),(0,0,1)]),
        }
        verts = []
        for face, (normal, faceVerts) in faces.items():
            neighbor = {
                "front":  (x,   y,   z+1),
                "back":   (x,   y,   z-1),
                "left":   (x-1, y,   z),
                "right":  (x+1, y,   z),
                "top":    (x,   y+1, z),
                "bottom": (x,   y-1, z)
            }[face]

            # If neighbor doesn't exist (or out of chunk) => face is visible
            if neighbor not in self.voxels:
                for vx, vy, vz in faceVerts:
                    verts.extend([
                        vx + x, vy + y, vz + z,
                        normal[0], normal[1], normal[2]
                    ])
        return np.array(verts, dtype='f4')

    def _update_voxel_and_neighbors(self, pos):
        """Rebuild mesh for pos and its neighbors (only)."""
        # Rebuild the mesh data for 'pos'
        self._voxel_mesh_data.pop(pos, None)  # remove old data
        if pos in self.voxels:
            self._voxel_mesh_data[pos] = self._build_voxel_geometry(pos)

        # For each neighbor, remove old data and regenerate
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        for ox, oy, oz in offsets:
            nx, ny, nz = pos[0]+ox, pos[1]+oy, pos[2]+oz
            neighbor = (nx, ny, nz)
            if neighbor in self._voxel_mesh_data:
                # Remove old geometry
                self._voxel_mesh_data.pop(neighbor, None)
                # If neighbor voxel still exists, rebuild its faces
                if neighbor in self.voxels:
                    self._voxel_mesh_data[neighbor] = self._build_voxel_geometry(neighbor)
        self._update_full_mesh()

    def _update_full_mesh(self):
        """Combine all voxel mesh data into a single array."""
        combined = []
        for verts in self._voxel_mesh_data.values():
            combined.extend(verts)
        self._mesh_cache = np.array(combined, dtype="f4")

    def generate_mesh(self):
        # Just return our combined mesh
        return self._mesh_cache

    def get_voxel_positions(self):
        return list(self.voxels.keys())

    def get_boundary_voxel_positions(self):
        boundary_positions = []
        for (x, y, z) in self.voxels.keys():
            if (x == 0 or y == 0 or z == 0 or
                x == self.WIDTH - 1 or y == self.HEIGHT - 1 or z == self.DEPTH - 1):
                boundary_positions.append((x, y, z))
        return boundary_positions