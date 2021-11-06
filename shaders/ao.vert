#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#include "cpu_gpu_shared_config.h"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in ivec4 inParticlePosition;
layout(location = 2) in float inParticleRadius;

layout(location = 0) out vec3 outSpherePosVS;
layout(location = 1) out float outSphereRadiusVS;

layout(set = 0, binding = 0) uniform camera_data
{
	/** Camera's view matrix */
	mat4 mViewMatrix;
	/** Camera's projection matrix */
	mat4 mProjMatrix;
} camData;

// ------------------ push constants ------------------
layout(push_constant) uniform PushConstants {
	mat4  mFragToVS;
	float mParticleRenderScale;
};
// ----------------------------------------------------

void main() {
	vec3 translation = vec3(inParticlePosition) / POS_RESOLUTION;
	float radius = inParticleRadius * mParticleRenderScale;
	float scale  = radius * 5.0;

	vec3 posWS  = inPosition * scale + translation;
	vec4 posVS  = camData.mViewMatrix * vec4(posWS, 1.0);
	gl_Position = camData.mProjMatrix * posVS;
	
	outSpherePosVS    = (camData.mViewMatrix * vec4(translation, 1.0)).xyz;
	outSphereRadiusVS = radius;
}
