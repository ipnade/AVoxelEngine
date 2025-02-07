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