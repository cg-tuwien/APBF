// ----------------------- PBD ------------------------
#define DIMENSIONS 2 // currently only supporting 2 or 3

#define SCENE 0 // 0: pool, 1: spherical_pool, 2: waterfall, 3: waterdrop
#define NEIGHBORHOOD_TYPE 3 // 0: brute force, 1: Green, 2: RTX, 3: binary search

#define POS_RESOLUTION 262144.0f
#define KERNEL_WIDTH_RESOLUTION 262144.0f
#define INCOMPRESSIBILITY_DATA_RESOLUTION 262144.0f
#define BOUNDARINESS_RESOLUTION 262144.0f

#define FIXED_TIME_STEP (1.0f / 60.0f) // 0 for variable timestep

#define KERNEL_SCALE 4.0f
#define KERNEL_WIDTH_PROPAGATION_FACTOR 0.5f

#define MAX_TRANSFERS 10000
// ----------------------------------------------------





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
	int mBoundarinessCalculationMethod;
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
