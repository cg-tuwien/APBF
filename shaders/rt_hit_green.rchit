#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 3, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec3 attribs;
layout(location = 2) rayPayloadEXT float secondaryRayHitValue;

void main()
{
//    vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
//    vec3 direction = normalize(vec3(0.8, 1, 0));
//    uint rayFlags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
//    uint cullMask = 0xff;
//    float tmin = 0.001;
//    float tmax = 100.0;
//
//    traceRayEXT(topLevelAS, rayFlags, cullMask, 1 /* sbtRecordOffset */, 0 /* sbtRecordStride */, 1 /* missIndex */, origin, tmin, direction, tmax, 2 /*payload location*/);

	hitValue = vec3(0, 1, 0);
}