# Python
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, MOUSEMOTION, MOUSEBUTTONDOWN, KEYDOWN, K_ESCAPE
import moderngl
import glm

from player import Player
from voxel_chunk import Chunk
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
        self.chunk = Chunk()
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
                hit_voxel, hit_normal = ray_voxel_traversal(ray_origin, ray_dir, self.chunk)
                if hit_voxel is not None:
                    if event.button == 1:  # Left click
                        self.chunk.remove_voxel(hit_voxel)
                    elif event.button == 3:  # Right click
                        if hit_normal is not None:
                            new_pos = (
                                hit_voxel[0] + int(hit_normal.x),
                                hit_voxel[1] + int(hit_normal.y),
                                hit_voxel[2] + int(hit_normal.z)
                            )
                            self.chunk.add_voxel(new_pos)
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
        
        # Convert matrices to column-major order for OpenGL
        mvp = np.array(proj * view, dtype='f4').transpose()
        # Change this line:
        self.shader_program.main_prog["mvp"].write(mvp.tobytes())

        # Update mesh only if the chunk has been modified.
        if self.chunk.needs_update:
            mesh_data = self.chunk.generate_mesh()
            self.mesh_vbo.orphan(size=mesh_data.nbytes)
            self.mesh_vbo.write(mesh_data.tobytes())
            self.chunk.needs_update = False

        # Ray cast to find looked-at voxel
        ray_origin = self.player.position
        ray_dir = self.player.get_direction()
        self.looked_at_voxel, _ = ray_voxel_traversal(ray_origin, ray_dir, self.chunk)

    def render(self):
        self.ctx.clear(0.2, 0.3, 0.4)
        self.mesh_vao.render(mode=moderngl.TRIANGLES)

        # Render outline if looking at a voxel
        if self.looked_at_voxel is not None:
            self.ctx.depth_func = '<='
            self.shader_program.outline_prog["mvp"].write(
                self.shader_program.main_prog["mvp"].read()
            )
            self.shader_program.outline_prog["voxel_pos"].value = self.looked_at_voxel
            self.shader_program.outline_vao.render(moderngl.LINES)
            self.ctx.depth_func = '<'  # Reset to normal depth testing

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