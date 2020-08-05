#pragma once

#include "velocity_handling.h"
#include "box_collision.h"
#include "neighborhood_brute_force.h"
#include "inter_particle_collision.h"
#include "incompressibility.h"

class pool
{
public:
	pool(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius = 1.0f);
	void update(float aDeltaTime);
	pbd::particles& particles();

private:
	pbd::particles mParticles;
	pbd::fluid mFluid;
	// mNeighbors is a list of index lists; the current gpu_list framework doesn't support
	// automatic index updates for this structure, so don't even bother linking it to mParticles.
	// Just make sure that particle add/delete/reorder doesn't happen between write and read, so that the indices are not outdated.
	pbd::gpu_list<sizeof(uint32_t) * 64> mNeighbors;
	pbd::gpu_list<sizeof(uint32_t) * 64> mNeighborsFluid;

	pbd::velocity_handling mVelocityHandling;
	pbd::box_collision mBoxCollision;
	pbd::neighborhood_brute_force mNeighborhoodCollision;
	pbd::neighborhood_brute_force mNeighborhoodFluid;
	pbd::inter_particle_collision mInterParticleCollision;
	pbd::incompressibility mIncompressibility;
};
