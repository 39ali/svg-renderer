in vec3 position;
in vec2 uv;

// TODO : use uniform block for instance data
// https://stackoverflow.com/questions/28444821/how-do-i-properly-declare-a-uniform-array-of-structs-in-glsl-so-that-i-can-point

in vec3 a_pos;
in vec2 a_scale;
in vec4 a_uv_data;
in vec2 a_fill_color;

// linear gradient in
in vec4 a_linear_gradient_bbox;
in vec4 a_stop_offsets;
in vec4 a_stop_colors;
in vec4 a_stop_alphas;
in int a_stop_count;
//

out vec2 uvs;
out vec2 texCoord;
out vec4 fill_color;

// linear gradient out
out vec4 linear_gradient_bbox;
out vec4 stop_offsets;
out vec4 stop_colors;
out vec4 stop_alphas;
flat out int stop_count;
//

uniform mat4 vp;

const float inv = 1 / 255.0;
vec4 color_to_vec4(vec2 color_alpha) {

  int color = int(color_alpha.x);
  int r = color & 0xff;
  int g = (color >> 8) & 0xff;
  int b = (color >> 16) & 0xff;

  vec3 c_out = vec3(float(r) * inv, float(g) * inv, float(b) * inv);

  return vec4(c_out, color_alpha.y);
}

void main() {

  vec3 pos = vec3(position.xy * a_scale, 0) + a_pos;

  //  pos.xy*=0.5;

  gl_Position = vp * vec4(pos, 1.0);

  texCoord = vec2(uv.x * a_uv_data.x + a_uv_data.y,
                  (1.0 - uv.y) * a_uv_data.z + a_uv_data.w);

  fill_color = color_to_vec4(a_fill_color);
  linear_gradient_bbox = a_linear_gradient_bbox;
  stop_offsets = a_stop_offsets;
  stop_colors = a_stop_colors;
  stop_alphas = a_stop_alphas;
  stop_count = a_stop_count;
  uvs = vec2(uv.x, 1.0 - uv.y);
}