#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 inParticlePositionRadius;
layout(location = 1) in int  inCustomIndexAndMask;

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
	float scale = inParticlePositionRadius[3];

	vec3 posWS = translation;
    gl_Position = appData.mProjMatrix * appData.mViewMatrix * vec4(posWS, 1.0);
	outColor = maskToColor();
}
