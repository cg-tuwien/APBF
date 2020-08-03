#version 460
#extension GL_ARB_separate_shader_objects : enable

#define POS_RESOLUTION 262144.0f

layout(location = 0) in vec3 inPosition;
layout(location = 1) in ivec4 inParticlePosition;
layout(location = 2) in float inParticleRadius;

layout(set = 0, binding = 0) uniform application_data
{
	mat4 mViewMatrix;
	mat4 mProjMatrix;
	vec4 mTime;
} appData;

void main() {
	vec3 translation = vec3(inParticlePosition) / POS_RESOLUTION;
	float scale = inParticleRadius;

	vec3 posWS = inPosition * scale + translation;
    gl_Position = appData.mProjMatrix * appData.mViewMatrix * vec4(posWS, 1.0);
}
