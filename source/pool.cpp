#include "pool.h"
#include "initialize.h"
#include "measurements.h"

pool::pool(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius) :
	mParticles(100000)
{
	shader_provider::start_recording();
	mParticles.request_length(100000);
	mFluid.request_length(100000);
	mNeighbors.request_length(100000);
	mNeighborsFluid.request_length(100000);
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_box_shape(mParticles, aMin + glm::vec3(2, 20, 2), aMax - glm::vec3(2, 2, 2), aRadius);
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	mVelocityHandling.add_particles(mParticles, glm::vec3(0, -10, 0));
	mBoxCollision.add_particles(mParticles);
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z));
	mInterParticleCollision.set_data(&mParticles, &mNeighbors);
	mIncompressibility.set_data(&mParticles, &mNeighborsFluid);
	mNeighborhoodCollision.set_data(&mParticles, &mParticles.hidden_list().get<pbd::hidden_particles::id::radius>(), &mNeighbors);
	mNeighborhoodFluid.set_data(&mParticles, &mParticles.hidden_list().get<pbd::hidden_particles::id::radius>(), &mNeighborsFluid);
//	mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::target_radius>(), &mNeighborsFluid);
	mNeighborhoodCollision.set_range_scale(2.0f);
	mNeighborhoodFluid.set_range_scale(4.0f);
	shader_provider::end_recording();
}

void pool::update(float aDeltaTime)
{
	mVelocityHandling.apply(aDeltaTime);
	mBoxCollision.apply();
	measurements::record_timing_interval_start("Neighborhood");
//	mNeighborhoodCollision.apply();
	mNeighborhoodFluid.apply();
	measurements::record_timing_interval_end("Neighborhood");
//	mInterParticleCollision.apply();
	mIncompressibility.apply();
}

pbd::particles& pool::particles()
{
	return mParticles;
}
