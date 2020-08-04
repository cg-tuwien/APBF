#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#include "cpu_gpu_shared_config.h"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inParticlePositionRadius;
layout(location = 2) in int  inCustomIndexAndMask;

layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform application_data
{
	/** Camera's view matrix */
	mat4 mViewMatrix;
	/** Camera's projection matrix */
	mat4 mProjMatrix;
	/** [0]: time since start, [1]: delta time, [2]: reset particle positions, [3]: set uniform particle radius  */
	vec4 mTimeAndUserInput;
	/** [0]: cullMask for traceRayEXT, [1]: neighborhood-origin particle-id, [2]: perform sphere intersection, [3]: unused  */
	uvec4 mUserInput;
} appData;

vec3 maskToColor()
{
	return vec3((inCustomIndexAndMask >> 24) & 0xF, (inCustomIndexAndMask >> 25) & 0xF, (inCustomIndexAndMask >> 26) & 0xF);
}

void main() {
	vec3 translation = inParticlePositionRadius.xyz;
	const float setUniformRadius = appData.mTimeAndUserInput[3];
	float scale = mix(inParticlePositionRadius[3], float(UNIFORM_PARTICLE_RADIUS), setUniformRadius);

	vec3 posWS = inPosition * scale + translation;
    gl_Position = appData.mProjMatrix * appData.mViewMatrix * vec4(posWS, 1.0);
	outColor = maskToColor();
}
