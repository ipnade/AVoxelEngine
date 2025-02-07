import moderngl
import numpy as np

class ShaderProgram:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.main_prog = self._create_main_program()
        self.outline_prog = self._create_outline_program()
        self._setup_outline_buffer()
        
    def _read_shader_file(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()
            
    def _create_main_program(self):
        prog = self.ctx.program(
            vertex_shader=self._read_shader_file('shaders/shader.vert'),
            fragment_shader=self._read_shader_file('shaders/shader.frag')
        )
        
        # Set up default uniforms
        prog["light_dir"].value = tuple(
            (np.array([-0.5, -1.0, -0.3]) / np.linalg.norm([-0.5, -1.0, -0.3])).tolist()
        )
        prog["object_color"].value = (0.5, 0.5, 0.5, 1.0)
        prog["ambient_color"].value = (0.2, 0.2, 0.2, 1.0)
        
        return prog
        
    def _create_outline_program(self):
        return self.ctx.program(
            vertex_shader=self._read_shader_file('shaders/outline.vert'),
            fragment_shader=self._read_shader_file('shaders/outline.frag')
        )
        
    def _setup_outline_buffer(self):
        outline_vertices = np.array([
            # Front face (CCW)
            0,0,1, 1,0,1,
            1,0,1, 1,1,1,
            1,1,1, 0,1,1,
            0,1,1, 0,0,1,
            
            # Back face (CW to be culled)
            1,0,0, 0,0,0,
            1,1,0, 1,0,0,
            0,1,0, 1,1,0,
            0,0,0, 0,1,0,
            
            # Side edges (front to back)
            0,0,1, 0,0,0,
            1,0,1, 1,0,0,
            1,1,1, 1,1,0,
            0,1,1, 0,1,0
        ], dtype='f4')

        outline_vbo = self.ctx.buffer(outline_vertices)
        self.outline_vao = self.ctx.vertex_array(
            self.outline_prog, 
            [(outline_vbo, '3f', 'in_position')]
        )
        
    def create_mesh_vao(self, mesh_vbo):
        return self.ctx.vertex_array(
            self.main_prog, 
            [(mesh_vbo, "3f 3f", "in_position", "in_normal")]
        )