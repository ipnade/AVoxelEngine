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

    # --- Create shaders ---
    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec3 in_normal;
            in vec3 instance_translation;
            uniform mat4 mvp;
            out vec3 v_normal;
            out vec3 v_frag_pos;
            void main() {
                vec4 world_pos = vec4(in_position + instance_translation, 1.0);
                gl_Position = mvp * world_pos;
                v_frag_pos = world_pos.xyz;
                // Note: assuming model matrix is identity so that normals stay in world space
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
                // Normalize the normal vector
                vec3 norm = normalize(v_normal);
                // Ambient term
                vec3 ambient = ambient_color.rgb * object_color.rgb;
                // Diffuse term
                float diff = max(dot(norm, -light_dir), 0.0);
                vec3 diffuse = diff * object_color.rgb;
                vec3 result = ambient + diffuse;
                fragColor = vec4(result, object_color.a);
            }
        """,
    )

    # Set lighting uniforms:
    prog["light_dir"].value = tuple((np.array([-0.5, -1.0, -0.3]) / np.linalg.norm([-0.5, -1.0, -0.3])).tolist())
    prog["object_color"].value = (0.5, 0.5, 0.5, 1.0)  # Gray color for stone voxels
    prog["ambient_color"].value = (0.2, 0.2, 0.2, 1.0)

    # --- Cube geometry (each cube is 1x1x1) ---
    cube_vertices = np.array([
        # front face (normal: 0,0,1)
        0,0,1,  0,0,1,    1,0,1,  0,0,1,    1,1,1,  0,0,1,
        0,0,1,  0,0,1,    1,1,1,  0,0,1,    0,1,1,  0,0,1,
        # back face (normal: 0,0,-1)
        1,0,0,  0,0,-1,   0,0,0,  0,0,-1,   0,1,0,  0,0,-1,
        1,0,0,  0,0,-1,   0,1,0,  0,0,-1,   1,1,0,  0,0,-1,
        # left face (normal: -1,0,0)
        0,0,0,  -1,0,0,   0,0,1,  -1,0,0,   0,1,1,  -1,0,0,
        0,0,0,  -1,0,0,   0,1,1,  -1,0,0,   0,1,0,  -1,0,0,
        # right face (normal: 1,0,0)
        1,0,1,  1,0,0,    1,0,0,  1,0,0,    1,1,0,  1,0,0,
        1,0,1,  1,0,0,    1,1,0,  1,0,0,    1,1,1,  1,0,0,
        # top face (normal: 0,1,0)
        0,1,1,  0,1,0,    1,1,1,  0,1,0,    1,1,0,  0,1,0,
        0,1,1,  0,1,0,    1,1,0,  0,1,0,    0,1,0,  0,1,0,
        # bottom face (normal: 0,-1,0)
        0,0,0,  0,-1,0,   1,0,0,  0,-1,0,   1,0,1,  0,-1,0,
        0,0,0,  0,-1,0,   1,0,1,  0,-1,0,   0,0,1,  0,-1,0
    ], dtype='f4')

    # Create an empty instance buffer BEFORE creating the VAO.
    instance_buffer = ctx.buffer(reserve=10000 * 12)  # reserve bytes for 10000 instances (3 floats each)

    vbo = ctx.buffer(cube_vertices.tobytes())

    vao = ctx.vertex_array(
        prog,
        [
            (vbo, "3f 3f", "in_position", "in_normal"),
            (instance_buffer, "3f/i", "instance_translation"),
        ]
    )

    # --- Create outline shader and geometry ---
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
                fragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black color
            }
        """,
    )

    # Define cube corners (for a unit cube from 0 to 1) and edge indices
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
        0,1, 1,2, 2,3, 3,0,  # bottom square
        4,5, 5,6, 6,7, 7,4,  # top square
        0,4, 1,5, 2,6, 3,7   # vertical edges
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

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick() / 1000.0  # Unlock framerate (no frame cap)

        # Process events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                player.process_mouse(dx, dy)
            elif event.type == MOUSEBUTTONDOWN:
                # Use ray intersection to remove/add voxels when clicking
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
                    if event.button == 1:  # Left click: remove voxel
                        chunk.remove_voxel(hit_voxel)
                    elif event.button == 3:  # Right click: add voxel adjacent to hit face
                        new_pos = (hit_voxel[0] + int(hit_normal[0]),
                                   hit_voxel[1] + int(hit_normal[1]),
                                   hit_voxel[2] + int(hit_normal[2]))
                        chunk.add_voxel(new_pos)

        keys = pygame.key.get_pressed()
        player.process_keyboard(keys, dt)

        # Calculate view and projection matrices
        view = Matrix44.look_at(
            player.position,
            player.position + player.get_direction(),
            Vector3([0.0, 1.0, 0.0]),
        )
        proj = Matrix44.perspective_projection(70.0, width / height, 0.1, 1000.0)
        mvp = proj * view
        prog["mvp"].write(mvp.astype("f4").tobytes())

        # Update instance buffer with all cube translations from the chunk
        positions = np.array(chunk.get_voxel_positions(), dtype='f4')
        if positions.size == 0:
            instance_buffer.write(np.array([], dtype='f4').tobytes())
        else:
            instance_buffer.write(positions.tobytes())

        # Render scene
        ctx.clear(0.2, 0.3, 0.4)
        vao.render(mode=moderngl.TRIANGLES, instances=len(chunk.get_voxel_positions()))

        # --- Determine the voxel being looked at (center ray) ---
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
            # Build model matrix for the outline (translate to hit voxel)
            model = Matrix44.from_translation(np.array(hit_voxel, dtype='f4'))
            outline_mvp = mvp * model
            outline_prog["mvp"].write(outline_mvp.astype("f4").tobytes())
            ctx.line_width = 2.0  # Set a thin line width
            outline_vao.render(mode=moderngl.LINES)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()