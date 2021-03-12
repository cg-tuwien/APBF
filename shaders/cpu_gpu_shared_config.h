// -------- neighborhood RTX proof of concept ---------
#define NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT 0

#define BLAS_CENTRIC 0
#if !BLAS_CENTRIC
#define INST_CENTRIC 1
#else
#define INST_CENTRIC 0
#endif

#define NOT_NEIGHBOR_MASK  0x01
#define NEIGHBOR_MASK      0x02
#define ORIGIN_MASK        0x06

#define RTX_NEIGHBORHOOD_RADIUS_FACTOR 30

#define UNIFORM_PARTICLE_RADIUS 0.07
// ----------------------------------------------------






// ----------------------- PBD ------------------------
#define DIMENSIONS 3 // currently only supporting 2 or 3

#define POS_RESOLUTION 262144.0
#define KERNEL_WIDTH_RESOLUTION 262144.0
#define INCOMPRESSIBILITY_DATA_RESOLUTION 262144.0
#define BOUNDARINESS_RESOLUTION 262144.0
#define NEIGHBOR_LIST_MAX_LENGTH 2048                            // TODO delete

#define FIXED_TIME_STEP (1.0f / 60.0f) // 0 for variable timestep

#define KERNEL_SCALE 4.0
#define KERNEL_SPREADING_FACTOR 0.5

#define MAX_TRANSFERS 100
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
	int mUpdateTargetRadius;
	int mUpdateBoundariness;
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
