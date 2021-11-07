#define NEIGHBOR_LIST_MAX_LENGTH 10000000
#define DIMENSIONS 3
//#define RTX_ACCELERATION_STRUCTURE_REBUILD_INTERVAL 60
#define RTX_ACCELERATION_STRUCTURE_REBUILD_INTERVAL 0

#define POS_RESOLUTION 262144.0f

#define PARTICLE_COUNT 50000
#define AREA_MIN glm::vec3(0, 0, 0)
#define AREA_MAX glm::vec3(80, 80, 80)
#define MIN_RADIUS 1.0f
#define MAX_RADIUS 10.0f
#define GREEN_RESOLUTION_LOG_2 4






#define NEIGHBORHOOD_TYPE 3 // 0: brute force, 1: Green, 2: RTX, 3: binary search





// ---------------------- HELPERS ---------------------
#define CONCAT_AUX(a, b) a ## b
#define CONCAT(a, b) CONCAT_AUX(a, b)
#define STRINGIZE_AUX(a) #a
#define STRINGIZE(a) STRINGIZE_AUX(a)
// ----------------------------------------------------





// ---------------------- SCENES ----------------------
#define SCENE_0 pool
#define SCENE_1 spherical_pool
#define SCENE_2 waterfall
#define SCENE_3 waterdrop

#define SCENE_NAME CONCAT(SCENE_, SCENE)
#define SCENE_FILENAME STRINGIZE(CONCAT(SCENE_NAME, .h))
// ----------------------------------------------------





// ----------------- NEIGHBOR SEARCH ------------------
#define NEIGHBOR_SEARCH_0 neighborhood_brute_force
#define NEIGHBOR_SEARCH_1 neighborhood_green
#define NEIGHBOR_SEARCH_2 neighborhood_rtx
#define NEIGHBOR_SEARCH_3 neighborhood_binary_search

#define NEIGHBOR_SEARCH_NAME CONCAT(NEIGHBOR_SEARCH_, NEIGHBORHOOD_TYPE)
#define NEIGHBOR_SEARCH_FILENAME STRINGIZE(CONCAT(NEIGHBOR_SEARCH_NAME, .h))
// ----------------------------------------------------





// ---------------------- structs ---------------------
#ifndef APBF_SETTINGS
#define APBF_SETTINGS

struct apbf_settings
{
	int mHeightKernelId;
	int mGradientKernelId;
	int mMerge;
	int mSplit;
	int mBaseKernelWidthOnTargetRadius;
	int mBaseKernelWidthOnBoundaryDistance;
	int mUpdateTargetRadius;
	int mUpdateBoundariness;
	int mNeighborListSorted;
	int mGroundTruthBoundaryDistance;
	float mBoundarinessAdaptionSpeed;
	float mKernelWidthAdaptionSpeed;
	float mBoundarinessSelfGradLengthFactor;
	float mBoundarinessUnderpressureFactor;
	float mNeighborBoundarinessThreshold;
	float mMergeDuration;
	float mSmallestTargetRadius;
	float mTargetRadiusOffset;
	float mTargetRadiusScaleFactor;
};
#endif
// ----------------------------------------------------
