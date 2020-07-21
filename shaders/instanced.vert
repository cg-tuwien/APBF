#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inParticlePositionRadius;

layout(set = 0, binding = 0) uniform application_data
{
	mat4 mViewMatrix;
	mat4 mProjMatrix;
	vec4 mTime;
} appData;

void main() {
	vec3 translation = inParticlePositionRadius.xyz;
	float scale = inParticlePositionRadius[3];

	vec3 posWS = inPosition * scale + translation;
    gl_Position = appData.mProjMatrix * appData.mViewMatrix * vec4(posWS, 1.0);
}
