#version 330

in vec3 in_position;

uniform mat4 mvp;
uniform vec3 voxel_pos;

void main() {
    vec3 world_pos = in_position + voxel_pos;
    vec4 pos = mvp * vec4(world_pos, 1.0);
    
    // Small offset towards camera to prevent z-fighting
    
    pos.z -= 0.0001;
    gl_Position = pos;
}