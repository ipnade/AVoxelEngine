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

# --- Voxel chunk ---
class Chunk:
    SIZE = 16

    def __init__(self):
        # voxels stored as keys with a value representing the block type ("Stone")
        self.voxels = {}
        # Create a full chunk (all voxels present, type "Stone")
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                for z in range(self.SIZE):
                    self.voxels[(x, y, z)] = "Stone"

    def remove_voxel(self, pos):
        if pos in self.voxels:
            del self.voxels[pos]

    def add_voxel(self, pos):
        self.voxels[pos] = "Stone"

    def get_voxel_positions(self):
        return list(self.voxels.keys())

    def generate_mesh(self):
        vertices = []
        # Define faces: each with a normal and the two triangles (6 vertices) as positions (relative to voxel)
        faces = {
            "front":  ((0,0,1),  [(0,0,1), (1,0,1), (1,1,1), (0,0,1), (1,1,1), (0,1,1)]),
            "back":   ((0,0,-1), [(1,0,0), (0,0,0), (0,1,0), (1,0,0), (0,1,0), (1,1,0)]),
            "left":   ((-1,0,0), [(0,0,0), (0,0,1), (0,1,1), (0,0,0), (0,1,1), (0,1,0)]),
            "right":  ((1,0,0),  [(1,0,1), (1,0,0), (1,1,0), (1,0,1), (1,1,0), (1,1,1)]),
            "top":    ((0,1,0),  [(0,1,1), (1,1,1), (1,1,0), (0,1,1), (1,1,0), (0,1,0)]),
            "bottom": ((0,-1,0), [(0,0,0), (1,0,0), (1,0,1), (0,0,0), (1,0,1), (0,0,1)]),
        }
        for pos in self.voxels.keys():
            x, y, z = pos
            for face, (normal, verts) in faces.items():
                # Determine neighbor coordinate based on face
                if face == "front":
                    neighbor = (x, y, z+1)
                elif face == "back":
                    neighbor = (x, y, z-1)
                elif face == "left":
                    neighbor = (x-1, y, z)
                elif face == "right":
                    neighbor = (x+1, y, z)
                elif face == "top":
                    neighbor = (x, y+1, z)
                elif face == "bottom":
                    neighbor = (x, y-1, z)
                # Only add face if neighbor is missing (not in voxels)
                if neighbor not in self.voxels:
                    for vx, vy, vz in verts:
                        # Append position (offset by voxel pos) and the face normal (each 3 floats)
                        vertices.extend([vx + x, vy + y, vz + z, normal[0], normal[1], normal[2]])
        return np.array(vertices, dtype="f4")

# --- Player controller ---
class Player:
    def __init__(self):
        self.position = Vector3([20.0, 20.0, 20.0])
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 10.0  # units per second
        self.sensitivity = 0.1

    def get_direction(self):
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
    width, height = 1600, 900  # Updated window size
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)

    # --- Create shader for voxels ---
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

    prog["light_dir"].value = tuple((np.array([-0.5, -1.0, -0.3])/ np.linalg.norm([-0.5, -1.0, -0.3])).tolist())
    prog["object_color"].value = (0.5, 0.5, 0.5, 1.0)  # Gray (Stone) voxels
    prog["ambient_color"].value = (0.2, 0.2, 0.2, 1.0)

    # --- Create outline shader and geometry (unchanged) ---
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

    # Create a buffer for the voxel mesh (reserve an initial size)
    mesh_vbo = ctx.buffer(reserve=10 * 1024 * 1024)  # adjust size as needed
    # Create a VAO for the voxel mesh; vertices have 3f for position and 3f for normal
    mesh_vao = ctx.vertex_array(
        prog, [(mesh_vbo, "3f 3f", "in_position", "in_normal")]
    )

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick() / 1000.0  # Unlocked framerate

        # Process events ...
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                player.process_mouse(dx, dy)
            elif event.type == MOUSEBUTTONDOWN:
                ray_origin = player.position
                ray_dir = np.array(player.get_direction(), dtype='f4')
                hit_voxel = None
                hit_t = float('inf')
                hit_normal = None
                for pos in chunk.get_voxel_positions():
                    box_min = np.array(pos, dtype='f4')
                    box_max = box_min + 1.0
                    result = ray_box_intersection(np.array(ray_origin, dtype='f4'), ray_dir, box_min, box_max)
                    if result is not None:
                        t, normal = result
                        if 0 < t < hit_t:
                            hit_t = t
                            hit_voxel = pos
                            hit_normal = normal
                if hit_voxel is not None:
                    if event.button == 1:
                        chunk.remove_voxel(hit_voxel)
                    elif event.button == 3:
                        new_pos = (hit_voxel[0] + int(hit_normal[0]),
                                   hit_voxel[1] + int(hit_normal[1]),
                                   hit_voxel[2] + int(hit_normal[2]))
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

        # Regenerate the chunk mesh from visible faces and update the mesh VBO
        mesh_data = chunk.generate_mesh()
        mesh_vbo.orphan(size=mesh_data.nbytes)
        mesh_vbo.write(mesh_data.tobytes())

        # Render the voxel chunk (only outer faces are in mesh_data)
        ctx.clear(0.2, 0.3, 0.4)
        mesh_vao.render(mode=moderngl.TRIANGLES)

        # --- Determine the voxel being looked at ---
        ray_origin = np.array(player.position, dtype='f4')
        ray_dir = np.array(player.get_direction(), dtype='f4')
        hit_voxel = None
        hit_t = float('inf')
        for pos in chunk.get_voxel_positions():
            box_min = np.array(pos, dtype='f4')
            box_max = box_min + 1.0
            result = ray_box_intersection(ray_origin, ray_dir, box_min, box_max)
            if result is not None:
                t, _ = result
                if 0 < t < hit_t:
                    hit_t = t
                    hit_voxel = pos

        # --- Render outline if a voxel is hit ---
        if hit_voxel is not None:
            model = Matrix44.from_translation(np.array(hit_voxel, dtype='f4'))
            outline_mvp = mvp * model
            outline_prog["mvp"].write(outline_mvp.astype("f4").tobytes())
            ctx.line_width = 2.0
            outline_vao.render(mode=moderngl.LINES)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()