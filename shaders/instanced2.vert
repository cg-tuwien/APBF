#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#include "cpu_gpu_shared_config.h"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in ivec4 inParticlePosition;
layout(location = 2) in float inParticleRadius;
layout(location = 3) in float inFloatForColor;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormalVS;

layout(set = 0, binding = 0) uniform camera_data
{
	/** Camera's view matrix */
	mat4 mViewMatrix;
	/** Camera's projection matrix */
	mat4 mProjMatrix;
	/** [0]: time since start, [1]: delta time, [2]: reset particle positions, [3]: set uniform particle radius  */
	vec4 mTimeAndUserInput;
	/** [0]: cullMask for traceRayEXT, [1]: neighborhood-origin particle-id, [2]: perform sphere intersection, [3]: unused  */
	uvec4 mUserInput;
} camData;

// ------------------ push constants ------------------
layout(push_constant) uniform PushConstants {
	vec3  mColor1;
	float mColor1Float;
	vec3  mColor2;
	float mColor2Float;
	float mParticleRenderScale;
};
// ----------------------------------------------------

void main() {
	vec3 translation = vec3(inParticlePosition) / POS_RESOLUTION;
	float scale = inParticleRadius * mParticleRenderScale;

	vec3 posWS  = inPosition * scale + translation;
	gl_Position = camData.mProjMatrix * camData.mViewMatrix * vec4(posWS, 1.0);
	float a     = (inFloatForColor - mColor1Float) / (mColor2Float - mColor1Float);
	a           = min(1, max(0, a));
	outColor    = mix(mColor1, mColor2, a);
	outNormalVS = mat3(camData.mViewMatrix) * inPosition;
}
