#define NEIGHBOR_LIST_MAX_LENGTH 10000000
#define DIMENSIONS 3
//#define RTX_ACCELERATION_STRUCTURE_REBUILD_INTERVAL 60
#define RTX_ACCELERATION_STRUCTURE_REBUILD_INTERVAL 0
#define POS_RESOLUTION 262144.0f
//#define PARTICLE_MESH "assets/icosahedron.obj"
#define PARTICLE_MESH "assets/sphere.obj"

#define PARTICLE_COUNT 10000
#define AREA_MIN glm::vec3(-40, -40, -40)
#define AREA_MAX glm::vec3(40, 40, 40)
#define MIN_RADIUS 1.0f
#define MAX_RADIUS 10.0f
#define GREEN_RESOLUTION_LOG_2 4






// ---------------------- structs ---------------------
#ifndef GPU_SETTINGS
#define GPU_SETTINGS

struct gpu_settings
{
	int   mNeighborListSorted;                 // bool
	int   mShowFocusParticleNeighborhoodRange; // bool
	int   mFocusParticleId;
	float mParticleRenderScale;
};
#endif
// ----------------------------------------------------
