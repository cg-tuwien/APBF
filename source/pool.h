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
#include "time_machine.h"
#include "user_controlled_boxes.h"

#define NEIGHBORHOOD_TYPE 2 // 0 = brute force, 1 = Green, 2 = RTX

class pool
{
public:
	pool(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius = 1.0f);
	void update(float aDeltaTime);
	pbd::particles& particles();
	pbd::fluid& fluid();
	pbd::neighbors& neighbors();
	auto& time_machine() { return mTimeMachine; };
	void handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos);
	void render(const glm::mat4& aViewProjection);

	bool mRenderBoxes;

private:
	pbd::particles mParticles;
	pbd::fluid mFluid;
	pbd::transfers mTransfers;
	// mNeighborsFluid is a list of index pairs; the current gpu_list framework doesn't support
	// automatic index updates for this structure, so don't even bother linking it to mParticles.
	// Just make sure that particle add/delete/reorder doesn't happen between write and read, so that the indices are not outdated.
	pbd::neighbors mNeighborsFluid;

	float mDeltaTime;
	pbd::time_machine<pbd::particles, pbd::hidden_particles, pbd::particles,
		pbd::gpu_list<4>, pbd::gpu_list<4>, pbd::gpu_list<4>, pbd::gpu_list<4>,
		pbd::gpu_list<4>, pbd::particles, pbd::particles, float> mTimeMachine;

	pbd::velocity_handling mVelocityHandling;
	pbd::box_collision mBoxCollision;
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
	user_controlled_boxes mUcb;
};
