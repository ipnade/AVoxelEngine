# Python
import math
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, MOUSEMOTION, MOUSEBUTTONDOWN, KEYDOWN, K_ESCAPE
import moderngl
from pyrr import Matrix44, Vector3

# --- Helpers for ray-box intersection ---
def ray_box_intersection(ray_origin, ray_dir, box_min, box_max):
    tmin = (box_min - ray_origin) / ray_dir
    tmax = (box_max - ray_origin) / ray_dir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    t_near = np.max(t1)
    t_far = np.min(t2)
    if t_far < 0 or t_near > t_far:
        return None
    # Determine hit face normal based on which axis t_near comes from
    epsilon = 1e-4
    normal = np.zeros(3, dtype='f4')
    for i in range(3):
        if abs((box_min[i] - ray_origin[i]) - ray_dir[i]*t_near) < epsilon:
            normal[i] = -1
            break
        if abs((box_max[i] - ray_origin[i]) - ray_dir[i]*t_near) < epsilon:
            normal[i] = 1
            break
    return t_near, normal

def ray_voxel_traversal(ray_origin, ray_direction, chunk, max_distance=128):
    origin = np.array(ray_origin, dtype='f4')
    direction = np.array(ray_direction, dtype='f4')
    direction /= (np.linalg.norm(direction) + 1e-8)

    # Find the starting voxel
    x, y, z = map(int, origin)
    step_x = 1 if direction[0] >= 0 else -1
    step_y = 1 if direction[1] >= 0 else -1
    step_z = 1 if direction[2] >= 0 else -1

    # Ray parameter when crossing Voxel boundaries
    def t_scale(i, o, d, step):
        boundary = i + (1 if step > 0 else 0)
        return (boundary - o) / (d + 1e-8)

    tMaxX = t_scale(x, origin[0], direction[0], step_x)
    tMaxY = t_scale(y, origin[1], direction[1], step_y)
    tMaxZ = t_scale(z, origin[2], direction[2], step_z)
    
    tDeltaX = abs(1.0 / (direction[0] + 1e-8))
    tDeltaY = abs(1.0 / (direction[1] + 1e-8))
    tDeltaZ = abs(1.0 / (direction[2] + 1e-8))

    traveled = 0.0
    normal = np.zeros(3, dtype='f4')

    while traveled <= max_distance:
        if 0 <= x < Chunk.WIDTH and 0 <= y < Chunk.HEIGHT and 0 <= z < Chunk.DEPTH:
            if (x, y, z) in chunk.voxels:
                return (x, y, z), normal  # Return both hit position and normal

        if tMaxX < tMaxY and tMaxX < tMaxZ:
            traveled = tMaxX
            tMaxX += tDeltaX
            x += step_x
            normal[:] = [-step_x, 0, 0]
        elif tMaxY < tMaxZ:
            traveled = tMaxY
            tMaxY += tDeltaY
            y += step_y
            normal[:] = [0, -step_y, 0]
        else:
            traveled = tMaxZ
            tMaxZ += tDeltaZ
            z += step_z
            normal[:] = [0, 0, -step_z]

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

# --- Helper: Returns whether the player's view has changed significantly ---
def player_view_changed():
    # For demonstration purposes, always return True.
    # Replace with actual logic if needed.
    return True

# --- Voxel chunk ---
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

# --- Player controller ---
class Player:
    def __init__(self):
        # Set starting position above the chunk (chunk ranges x:0-15, y:0-63, z:0-15)
        self.position = Vector3([8.0, 70.0, 8.0])
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 10.0  # Units per second
        self.sensitivity = 0.1

    def get_direction(self):
        # Calculate forward vector
        # Calculate forward vector
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        x = math.cos(rad_yaw) * math.cos(rad_pitch)
        y = math.sin(rad_pitch)
        z = math.sin(rad_yaw) * math.cos(rad_pitch)
        return Vector3([x, y, z]).normalized

    def get_right(self):
        # Right vector is computed as cross(forward, world_up)
        return self.get_direction().cross(Vector3([0, 1, 0])).normalized

    def get_left(self):
        return -self.get_right()

    def get_up(self):
        # Cross product of right and forward gives up
        return self.get_right().cross(self.get_direction()).normalized

    def process_mouse(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def process_keyboard(self, keys, dt):
        forward = self.get_direction()
        right = self.get_right()
        left = self.get_left()  # Added left vector for clarity, though -right works too.
        up = Vector3([0, 1, 0])
        if keys[pygame.K_w]:
            self.position += forward * (self.speed * dt)
        if keys[pygame.K_s]:
            self.position -= forward * (self.speed * dt)
        if keys[pygame.K_a]:
            self.position += left * (self.speed * dt)
        if keys[pygame.K_d]:
            self.position += right * (self.speed * dt)
        if keys[pygame.K_SPACE]:
            self.position += up * (self.speed * dt)
        if keys[pygame.K_LSHIFT]:
            self.position -= up * (self.speed * dt)

# --- Main application class ---
def main():
    pygame.init()
    width, height = 1600, 900
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)

    # --- Create shader for voxels (unchanged) ---
    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec3 in_normal;
            uniform mat4 mvp;
            out vec3 v_normal;
            out vec3 v_frag_pos;
            void main() {
                vec4 world_pos = vec4(in_position, 1.0);
                gl_Position = mvp * world_pos;
                v_frag_pos = world_pos.xyz;
                v_normal = in_normal;
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_normal;
            in vec3 v_frag_pos;
            uniform vec3 light_dir;
            uniform vec4 object_color;
            uniform vec4 ambient_color;
            out vec4 fragColor;
            void main() {
                vec3 norm = normalize(v_normal);
                vec3 ambient = ambient_color.rgb * object_color.rgb;
                float diff = max(dot(norm, -light_dir), 0.0);
                vec3 diffuse = diff * object_color.rgb;
                fragColor = vec4(ambient + diffuse, object_color.a);
            }
        """,
    )

    prog["light_dir"].value = tuple(
        (np.array([-0.5, -1.0, -0.3]) / np.linalg.norm([-0.5, -1.0, -0.3])).tolist()
    )
    prog["object_color"].value = (0.5, 0.5, 0.5, 1.0)
    prog["ambient_color"].value = (0.2, 0.2, 0.2, 1.0)

    # --- Outline shader and geometry (unchanged) ---
    outline_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_position;
            uniform mat4 mvp;
            void main(){
                gl_Position = mvp * vec4(in_position, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            out vec4 fragColor;
            void main(){
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        """,
    )

    outline_vertices = np.array([
        0,0,0,
        1,0,0,
        1,1,0,
        0,1,0,
        0,0,1,
        1,0,1,
        1,1,1,
        0,1,1,
    ], dtype='f4')
    outline_indices = np.array([
        0,1, 1,2, 2,3, 3,0,  # bottom
        4,5, 5,6, 6,7, 7,4,  # top
        0,4, 1,5, 2,6, 3,7   # verticals
    ], dtype='i4')
    outline_vbo = ctx.buffer(outline_vertices.tobytes())
    outline_ibo = ctx.buffer(outline_indices.tobytes())
    outline_vao = ctx.vertex_array(
        outline_prog,
        [(outline_vbo, "3f", "in_position")],
        outline_ibo
    )

    # --- World and Player ---
    chunk = Chunk()
    player = Player()

    # Create a buffer for the voxel mesh
    mesh_vbo = ctx.buffer(reserve=10 * 1024 * 1024)
    mesh_vao = ctx.vertex_array(
        prog, [(mesh_vbo, "3f 3f", "in_position", "in_normal")]
    )

    clock = pygame.time.Clock()

    # --- Throttle outline update ---
    outline_update_interval = 0.1
    outline_timer = 0.0
    cached_hit_voxel = None

    running = True
    while running:
        dt = clock.tick() / 1000.0
        outline_timer += dt

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                player.process_mouse(dx, dy)
            elif event.type == MOUSEBUTTONDOWN:
                ray_origin = player.position
                ray_dir = np.array(player.get_direction(), dtype='f4')
                hit_voxel, hit_normal = ray_voxel_traversal(player.position, player.get_direction(), chunk)
                if hit_voxel is not None:
                    if event.button == 1:  # Left click
                        chunk.remove_voxel(hit_voxel)
                    elif event.button == 3:  # Right click
                        new_pos = (
                            hit_voxel[0] + int(hit_normal[0]),
                            hit_voxel[1] + int(hit_normal[1]),
                            hit_voxel[2] + int(hit_normal[2])
                        )
                        chunk.add_voxel(new_pos)

        keys = pygame.key.get_pressed()
        player.process_keyboard(keys, dt)

        view = Matrix44.look_at(
            player.position,
            player.position + player.get_direction(),
            Vector3([0.0, 1.0, 0.0]),
        )
        proj = Matrix44.perspective_projection(70.0, width/height, 0.1, 1000.0)
        mvp = proj * view
        prog["mvp"].write(mvp.astype("f4").tobytes())

        # Update mesh only if the chunk has been modified.
        if chunk.needs_update:
            mesh_data = chunk.generate_mesh()
            mesh_vbo.orphan(size=mesh_data.nbytes)
            mesh_vbo.write(mesh_data.tobytes())
            chunk.needs_update = False

        ctx.clear(0.2, 0.3, 0.4)
        mesh_vao.render(mode=moderngl.TRIANGLES)

        # Update outline only every outline_update_interval seconds:
        if outline_timer >= outline_update_interval:
            if player_view_changed():
                ray_origin = np.array(player.position, dtype='f4')
                ray_dir = np.array(player.get_direction(), dtype='f4')
                hit_voxel = None
                hit_t = float('inf')
                # Scan only nearby voxels instead of the full chunk boundary.
                for pos in get_nearby_boundary_voxels(chunk, player.position):
                    box_min = np.array(pos, dtype='f4')
                    box_max = box_min + 1.0
                    result = ray_box_intersection(ray_origin, ray_dir, box_min, box_max)
                    if result is not None:
                        t, _ = result
                        if 0 < t < hit_t:
                            hit_t = t
                            hit_voxel = pos
                cached_hit_voxel = hit_voxel
            outline_timer = 0.0

        # Render outline using cached_hit_voxel, if any:
        if cached_hit_voxel is not None:
            model = Matrix44.from_translation(np.array(cached_hit_voxel, dtype='f4'))
            outline_mvp = mvp * model
            outline_prog["mvp"].write(outline_mvp.astype("f4").tobytes())
            ctx.line_width = 2.0
            outline_vao.render(mode=moderngl.LINES)

        # Update window title with framerate counter and player coordinates:
        pygame.display.set_caption(
            "AVoxelEngine - FPS: {:.2f} - Pos: {:.2f}, {:.2f}, {:.2f}".format(
                clock.get_fps(), player.position.x, player.position.y, player.position.z
            )
        )

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()