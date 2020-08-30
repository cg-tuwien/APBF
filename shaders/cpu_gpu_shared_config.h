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
#define POS_RESOLUTION 262144.0
#define KERNEL_WIDTH_RESOLUTION 262144.0
#define NEIGHBOR_LIST_MAX_LENGTH 512
#define DIMENSIONS 3
#define ADAPTIVE_SAMPLING 0
#define PARTICLE_RENDER_SCALE 0.25

#define KERNEL_ID 1
#define KERNEL_SCALE 4.0
#define KERNEL_SPREADING_FACTOR 0.5

#define SMALLEST_TARGET_RADIUS 0.5
#define TARGET_RADIUS_OFFSET 4.0
#define TARGET_RADIUS_SCALE_FACTOR 0.1

#define MERGE_DURATION 2.0f
#define SPLIT_DURATION 2.0f
// ----------------------------------------------------





// ---------------------- structs ---------------------
#ifndef APBF_SETTINGS
#define APBF_SETTINGS

struct apbf_settings
{
	int mKernelId;
	int mMerge;
	int mSplit;
	int mBaseKernelWidthOnTargetRadius;
	int mUpdateTargetRadius;
	int mUpdateBoundariness;
};
#endif
// ----------------------------------------------------
