import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, MOUSEMOTION, MOUSEBUTTONDOWN, KEYDOWN, K_ESCAPE
import moderngl
import glm

from player import Player
from world import World  # Updated: use World instead of single Chunk
from raycast import ray_voxel_traversal
from shader_program import ShaderProgram

class VoxelEngine:
    def __init__(self, width=1600, height=900):
        self.width = width
        self.height = height
        self.init_pygame()
        self.init_opengl()
        self.shader_program = ShaderProgram(self.ctx)  # Create shader program
        self.init_game_objects()
        
    def init_pygame(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.clock = pygame.time.Clock()
        
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = 'ccw'
    
    def init_game_objects(self):
        self.world = World()  # New: spawn a 4x4 grid of chunks
        self.player = Player()
        self.mesh_vbo = self.ctx.buffer(reserve=10 * 1024 * 1024)
        self.mesh_vao = self.shader_program.create_mesh_vao(self.mesh_vbo)
        
    def process_events(self, dt):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False
            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                self.player.process_mouse(dx, dy)
            elif event.type == MOUSEBUTTONDOWN:
                ray_origin = self.player.position
                ray_dir = self.player.get_direction()
                hit_voxel, hit_normal = self.world.ray_voxel_traversal(ray_origin, ray_dir)
                if hit_voxel is not None:
                    chunk = self.world.get_chunk_at(hit_voxel)
                    if chunk:
                        # Calculate local voxel coordinate within the chunk.
                        local_hit = (
                            hit_voxel[0] - chunk.offset[0],
                            hit_voxel[1] - chunk.offset[1],
                            hit_voxel[2] - chunk.offset[2]
                        )
                        if event.button == 1:  # Left click: remove voxel
                            chunk.remove_voxel(local_hit)
                        elif event.button == 3:  # Right click: add voxel
                            if hit_normal is not None:
                                new_local = (
                                    local_hit[0] + int(hit_normal.x),
                                    local_hit[1] + int(hit_normal.y),
                                    local_hit[2] + int(hit_normal.z)
                                )
                                chunk.add_voxel(new_local)
        keys = pygame.key.get_pressed()
        self.player.process_keyboard(keys, dt)
        return True

    def update(self, dt):
        view = glm.lookAt(
            self.player.position,  # eye
            self.player.position + self.player.get_direction(),  # center
            glm.vec3(0.0, 1.0, 0.0)  # up
        )
        
        proj = glm.perspective(
            glm.radians(75.0),  # fov in radians
            self.width/self.height,  # aspect ratio
            0.1,  # near
            1000.0  # far
        )
        
        mvp = np.array(proj * view, dtype='f4').transpose()
        self.shader_program.main_prog["mvp"].write(mvp.tobytes())

        if self.world.any_chunk_needs_update():
            mesh_data = self.world.generate_mesh()
            self.mesh_vbo.orphan(size=mesh_data.nbytes)
            self.mesh_vbo.write(mesh_data.tobytes())
            self.world.clear_update_flags()

        ray_origin = self.player.position
        ray_dir = self.player.get_direction()
        self.looked_at_voxel, _ = self.world.ray_voxel_traversal(ray_origin, ray_dir)

    def render(self):
        self.ctx.clear(0.2, 0.3, 0.4)
        self.mesh_vao.render(mode=moderngl.TRIANGLES)
        if self.looked_at_voxel is not None:
            self.ctx.depth_func = '<='
            self.shader_program.outline_prog["mvp"].write(
                self.shader_program.main_prog["mvp"].read()
            )
            self.shader_program.outline_prog["voxel_pos"].value = self.looked_at_voxel
            self.shader_program.outline_vao.render(moderngl.LINES)
            self.ctx.depth_func = '<'

    def display_fps(self):
        pygame.display.set_caption(
            "AVoxelEngine - FPS: {:.2f} - Pos: {:.2f}, {:.2f}, {:.2f}".format(
                self.clock.get_fps(), self.player.position.x, self.player.position.y, self.player.position.z
            )
        )

    def run(self):
        running = True
        while running:
            dt = self.clock.tick() / 1000.0
            running = self.process_events(dt)
            self.update(dt)
            self.render()
            self.display_fps()
            pygame.display.flip()
        pygame.quit()

def main():
    engine = VoxelEngine()
    engine.run()

if __name__ == "__main__":
    main()