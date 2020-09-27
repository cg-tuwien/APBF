#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inBoxMin;
layout(location = 2) in vec4 inBoxMax;

// ------------------ push constants ------------------
layout(push_constant) uniform PushConstants {
	mat4 mViewProjection;
};
// ----------------------------------------------------

void main() {
	vec3 posWS  = inPosition * (inBoxMax.xyz - inBoxMin.xyz) + inBoxMin.xyz;
	gl_Position = mViewProjection * vec4(posWS, 1.0);
}
