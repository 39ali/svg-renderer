precision mediump float;

in vec2 uvs;
in vec2 texCoord;
in vec4 fill_color;

// linear gradient
in vec4 linear_gradient_bbox;
in vec4 stop_offsets;
in vec4 stop_colors;
in vec4 stop_alphas;
flat in int stop_count;
//

uniform sampler2D texture0;

const float pxRange = 8.0;
const vec4 CLEAR = vec4(0.0, 0.0, 0.0, 0.0);

//            const float thickness= 0.0; ; // : range(-1.0, +1.0);
//            const float border = 0.0; // : range(0.0, 0.25);
//            const  vec2 shadowVector = vec2(+0.0, -0.0); //  : range(-0.25,
//            +0.25); const float shadowSoftness = 0.0 ;//: range(0.0, 1.0);
//            const float shadowOpacity = 0.0 ;//: range(0.0, 1.0);

// const vec3 bottomColor = vec3(0.0, 0.0, 1.0);
// const vec3 borderColor = vec3(1.0,1.0,0.0);

out vec4 color;

const float inv = 1 / 255.0;
vec4 color_to_vec4(float c, float alpha) {

  int color = int(c);
  int r = color & 0xff;
  int g = (color >> 8) & 0xff;
  int b = (color >> 16) & 0xff;

  vec3 c_out = vec3(float(r) * inv, float(g) * inv, float(b) * inv);

  return vec4(c_out, alpha);
}

float linearStep(float a, float b, float x) {
  return clamp((x - a) / (b - a), 0.0, 1.0);
}

float project_point_to_line_segment(vec2 A, vec2 B, vec2 p) {

  vec2 AB = (B - A);
  float AB_squared = dot(AB, AB);

  vec2 Ap = (p - A);
  float t = dot(Ap, AB) / AB_squared;

  return t;
}

vec4 linear_gradient() {

  vec4 c = CLEAR;

  float x0 = linear_gradient_bbox.x;
  float x1 = linear_gradient_bbox.y;
  float y0 = linear_gradient_bbox.z;
  float y1 = linear_gradient_bbox.w;

  vec4 stop_colors_0 = color_to_vec4(stop_colors.x, stop_alphas.x);
  vec4 stop_colors_1 = color_to_vec4(stop_colors.y, stop_alphas.y);
  vec4 stop_colors_2 = color_to_vec4(stop_colors.z, stop_alphas.z);
  vec4 stop_colors_3 = color_to_vec4(stop_colors.w, stop_alphas.w);

  float t = project_point_to_line_segment(vec2(x0, y0), vec2(x1, y1), uvs);
  //#1-2
  {
    // only check if t <offset2
    float in_range = step(t, stop_offsets.y);
    float tc = linearStep(stop_offsets.x, stop_offsets.y, t);
    c += mix(CLEAR, mix(stop_colors_0, stop_colors_1, tc), in_range);
  }

  //#2-3
  {

    float in_range = step(stop_offsets.y, t) * step(t, stop_offsets.z);
    float tc = linearStep(stop_offsets.y, stop_offsets.z, t);
    c += mix(CLEAR, mix(stop_colors_1, stop_colors_2, tc), in_range);
  }

  //#3-4
  {
    // only check if t > offset1
    float in_range = step(stop_offsets.z, t);
    float tc = linearStep(stop_offsets.z, stop_offsets.w, t);
    c += mix(CLEAR, mix(stop_colors_2, stop_colors_3, tc), in_range);
  }

  return c;
}

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

float screenPxRange(vec2 p) {
  vec2 unitRange = vec2(pxRange) / vec2(textureSize(texture0, 0));
  vec2 screenTexSize = vec2(1.0) / fwidth(p);
  return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

void main() {

  vec4 final_color;
  if (stop_count > 0) {
    final_color = linear_gradient();
  } else {
    final_color = fill_color;
  }

  vec2 p = texCoord;
  vec3 msd = texture(texture0, p).rgb;
  float sd = median(msd.r, msd.g, msd.b);

  //  float w = fwidth( sd );
  //  float opacity = smoothstep( 0.5 - w, 0.5 + w, sd );

  float screenPxDistance = screenPxRange(p) * (sd - 0.5);
  float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);
  color = vec4(final_color.rgb, final_color.a * opacity);
}