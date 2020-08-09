#version 460
#extension GL_ARB_separate_shader_objects : enable

#define POS_RESOLUTION 262144.0f

layout(location = 0) in vec3 inPosition;
layout(location = 1) in ivec4 inParticlePosition;
layout(location = 2) in float inParticleRadius;
layout(location = 3) in float inBoundaryDistance;

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

void main() {
	vec3 translation = vec3(inParticlePosition) / POS_RESOLUTION;
	float scale = inParticleRadius;

	vec3 posWS = inPosition * scale + translation;
	gl_Position = appData.mProjMatrix * appData.mViewMatrix * vec4(posWS, 1.0);
	outColor = vec3(inBoundaryDistance / 10, 0.2, 0);
}
