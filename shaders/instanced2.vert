#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#include "cpu_gpu_shared_config.h"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in ivec4 inParticlePosition;
layout(location = 2) in float inParticleRadius;
layout(location = 3) in float inBoundariness;
layout(location = 4) in float inBoundaryDistance;

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

// ------------------ push constants ------------------
layout(push_constant) uniform PushConstants {
	int mColor;
};
// ----------------------------------------------------

void main() {
	vec3 translation = vec3(inParticlePosition) / POS_RESOLUTION;
	float scale = inParticleRadius * PARTICLE_RENDER_SCALE;

	vec3 posWS = inPosition * scale + translation;
	gl_Position = appData.mProjMatrix * appData.mViewMatrix * vec4(posWS, 1.0);
	switch (mColor) {
		case 0:  outColor = vec3(inBoundariness         , 0.2, 0); break;
		case 1:  outColor = vec3(inBoundaryDistance / 32, 0.2, 0); break;
		default: outColor = vec3(0, 0, 1);                         break;
	}
}
