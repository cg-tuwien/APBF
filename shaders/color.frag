#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in  vec3 inColor;
layout(location = 1) in  vec3 inNormalVS;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormalVS;

void main() {
    outColor = vec4(inColor, 1.0);
    outNormalVS = vec4(inNormalVS, 0.0);
}
