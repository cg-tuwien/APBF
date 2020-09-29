#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) flat in uint inSelected;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = inSelected != 0u ? vec4(1, 0.5, 0.5, 0.1) : vec4(0, 1, 1, 0.1);
}
