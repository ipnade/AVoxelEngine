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