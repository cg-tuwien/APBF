// Large portions of this code are copied from https://github.com/dli/fluid/blob/master/shaders/sphereao.frag
// Copyright (c) 2016 David Li (http://david.li) under the MIT License

#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.14159265

layout(set = 1, binding = 0) uniform texture2D uNormal;
layout(set = 1, binding = 1) uniform texture2D uDepth;

//layout(pixel_center_integer) in vec4 gl_FragCoord;

layout(location = 0) in vec3  inSpherePosVS;
layout(location = 1) in float inSphereRadiusVS;

layout(location = 0) out vec4 outColor;

// ------------------ push constants ------------------
layout(push_constant) uniform PushConstants {
	mat4  mFragToVS;
	float mParticleRenderScale;
};
// ----------------------------------------------------

void main() {
	vec3  normalVS = texelFetch(uNormal, ivec2(gl_FragCoord.xy), 0).xyz;
	float depth    = texelFetch(uDepth , ivec2(gl_FragCoord.xy), 0).x;
	if (depth == 1) discard;

	// reconstruct position from depth buffer
	vec4 viewSpace = mFragToVS * vec4(gl_FragCoord.xy, depth, 1);
	vec3 positionVS = viewSpace.xyz / viewSpace.w;

	vec3 di = inSpherePosVS - positionVS;
	float l = length(di);

	float nl = dot(normalize(normalVS), di / l);
	float h = l / inSphereRadiusVS;
	float h2 = h * h;
	float k2 = 1.0 - h2 * nl * nl;

	float result = max(0.0, nl) / h2;

	if (k2 > 0.0 && l > inSphereRadiusVS) {
		result = nl * acos(-nl * sqrt((h2 - 1.0) / (1.0 - nl * nl))) - sqrt(k2 * (h2 - 1.0));
		result = result / h2 + atan(sqrt(k2 / (h2 - 1.0)));
		result /= PI;

		//result = pow( clamp(0.5*(nl*h+1.0)/h2,0.0,1.0), 1.5 ); //cheap approximation
	}

	outColor = vec4(result, result, result, 1.0);
}
