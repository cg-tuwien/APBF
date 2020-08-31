#pragma once

#include "velocity_handling.h"
#include "box_collision.h"
#include "neighborhood_brute_force.h"
#include "neighborhood_green.h"
#include "neighborhood_rtx.h"
#include "inter_particle_collision.h"
#include "spread_kernel_width.h"
#include "incompressibility.h"
#include "update_transfers.h"
#include "particle_transfer.h"

#define NEIGHBORHOOD_TYPE 2 // 0 = brute force, 1 = Green, 2 = RTX

class pool
{
public:
	pool(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius = 1.0f);
	void update(float aDeltaTime);
	pbd::particles& particles();
	pbd::fluid& fluid();

private:
	pbd::particles mParticles;
	pbd::fluid mFluid;
	pbd::transfers mTransfers;
	// mNeighbors is a list of index lists; the current gpu_list framework doesn't support
	// automatic index updates for this structure, so don't even bother linking it to mParticles.
	// Just make sure that particle add/delete/reorder doesn't happen between write and read, so that the indices are not outdated.
	pbd::gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH> mNeighbors;
	pbd::gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH> mNeighborsFluid;

	pbd::velocity_handling mVelocityHandling;
	pbd::box_collision mBoxCollision;
	pbd::neighborhood_brute_force mNeighborhoodCollision;
	pbd::inter_particle_collision mInterParticleCollision;
	pbd::spread_kernel_width mSpreadKernelWidth;
	pbd::incompressibility mIncompressibility;
	pbd::update_transfers mUpdateTransfers;
	pbd::particle_transfer mParticleTransfer;
#if NEIGHBORHOOD_TYPE == 0
	pbd::neighborhood_brute_force mNeighborhoodFluid;
#elif NEIGHBORHOOD_TYPE == 1
	pbd::neighborhood_green mNeighborhoodFluid;
#else
	pbd::neighborhood_rtx mNeighborhoodFluid;
#endif
};
