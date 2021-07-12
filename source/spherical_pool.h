#pragma once

#include "../shaders/cpu_gpu_shared_config.h"
#include "velocity_handling.h"
#include "spread_kernel_width.h"
#include "sphere_collision.h"
#include "incompressibility.h"
#include "update_transfers.h"
#include "particle_transfer.h"
#include "save_particle_info.h"
#include "time_machine.h"
#include NEIGHBOR_SEARCH_FILENAME

class spherical_pool
{
public:
	spherical_pool(const glm::vec3& aCenter, float aPoolRadius, float aParticleRadius = 1.0f);
	spherical_pool(const glm::vec3& aCenter, float aPoolRadius, gvk::camera& aCamera, float aParticleRadius = 1.0f);
	void update(float aDeltaTime);
	pbd::particles& particles();
	pbd::fluid& fluid();
	pbd::neighbors& neighbors();
	auto& time_machine() { return mTimeMachine; };
	void handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos);
	void render(const glm::mat4& aViewProjection);

	pbd::gpu_list<4> scalar_particle_velocities();
	float max_expected_boundary_distance();

private:
	pbd::particles mParticles;
	pbd::fluid mFluid;
	pbd::transfers mTransfers;
	// mNeighborsFluid is a list of index pairs; the current gpu_list framework doesn't support
	// automatic index updates for this structure, so don't even bother linking it to mParticles.
	// Just make sure that particle add/delete/reorder doesn't happen between write and read, so that the indices are not outdated.
	pbd::neighbors mNeighborsFluid;

	glm::vec2 mViewBoxMin;
	glm::vec2 mViewBoxMax;
	float mMaxExpectedBoundaryDistance;
	float mDeltaTime;
	pbd::time_machine<pbd::particles, pbd::hidden_particles, pbd::particles,
		pbd::gpu_list<4>, pbd::gpu_list<4>, pbd::gpu_list<4>, pbd::gpu_list<4>,
		pbd::gpu_list<4>, pbd::particles, pbd::particles, float> mTimeMachine;

	pbd::velocity_handling mVelocityHandling;
	pbd::spread_kernel_width mSpreadKernelWidth;
	pbd::sphere_collision mSphereCollision;
	pbd::incompressibility mIncompressibility;
	pbd::update_transfers mUpdateTransfers;
	pbd::particle_transfer mParticleTransfer;
	pbd::save_particle_info mSaveParticleInfo;
	pbd::NEIGHBOR_SEARCH_NAME mNeighborhoodFluid;
};
