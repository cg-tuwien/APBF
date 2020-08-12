#include "pool.h"
#include "initialize.h"
#include "measurements.h"

pool::pool(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius) :
	mParticles(100000),
	mTransfers(100)
{
	shader_provider::start_recording();
	mParticles.request_length(100000);
	mFluid.request_length(100000);
//	mTransfers.request_length(100); // not even necessary because index list never gets used
	mNeighbors.request_length(100000);
	mNeighborsFluid.request_length(100000);
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_box_shape(mParticles, aMin + glm::vec3(2, 2, 2), aMax - glm::vec3(2, 4, 2), aRadius);
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.length(), 4, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::boundary_distance>().write().buffer(), mFluid.length(), 0, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::target_radius>().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence(mFluid.get<pbd::fluid::id::transferring>().write().buffer(), mFluid.length(), 0, 0);
	mVelocityHandling.add_particles(mParticles, glm::vec3(0, -10, 0));
	mBoxCollision.add_particles(mParticles);
	mBoxCollision.add_box(aMin, glm::vec3(aMin.x + 2, aMax.y, aMax.z));
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z));
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMax.y, aMin.z + 2));
	mBoxCollision.add_box(glm::vec3(aMax.x - 2, aMin.y, aMin.z), aMax);
	mBoxCollision.add_box(glm::vec3(aMin.x, aMax.y - 2, aMin.z), aMax);
	mBoxCollision.add_box(glm::vec3(aMin.x, aMin.y, aMax.z - 2), aMax);
	mInterParticleCollision.set_data(&mParticles, &mNeighbors);
	mIncompressibility.set_data(&mFluid, &mNeighborsFluid, &mTransfers);
	mParticleTransfer.set_data(&mFluid, &mTransfers);
	mNeighborhoodCollision.set_data(&mParticles, &mParticles.hidden_list().get<pbd::hidden_particles::id::radius>(), &mNeighbors);
	mNeighborhoodCollision.set_range_scale(2.0f);
//	mNeighborhoodFluid.set_data(&mParticles, &mParticles.hidden_list().get<pbd::hidden_particles::id::radius>(), &mNeighborsFluid);
	mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::kernel_width>(), &mNeighborsFluid);
#if NEIGHBORHOOD_TYPE == 1
	mNeighborhoodFluid.set_position_range(aMin, aMax, 6u);
#endif
	mNeighborhoodFluid.set_range_scale(1.5f);
	shader_provider::end_recording();
}

void pool::update(float aDeltaTime)
{
	mVelocityHandling.apply(aDeltaTime);
//	mParticleTransfer.apply(aDeltaTime);
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

pbd::fluid& pool::fluid()
{
	return mFluid;
}
