#include "pool.h"
#include "initialize.h"

pool::pool(const glm::vec3& aMin, const glm::vec3& aMax) :
	mParticles(1024)
{
	shader_provider::start_recording();
	mParticles.request_length(1024);
	mFluid.request_length(1024);
	mNeighbors.request_length(1024);
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_box_shape(mParticles, aMin + glm::vec3(2, 20, 2), aMax - glm::vec3(2, 2, 2));
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	mVelocityHandling.add_particles(mParticles, glm::vec3(0, -10, 0));
	mBoxCollision.add_particles(mParticles);
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z));
	mInterParticleCollision.set_data(&mParticles, &mNeighbors);
	mNeighborhoodCollision.set_data(&mParticles, &mParticles.hidden_list().get<pbd::hidden_particles::id::radius>(), &mNeighbors);
	//mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::target_radius>(), &mNeighborsFluid);
	mNeighborhoodCollision.set_range_scale(2.0f);
	shader_provider::end_recording();
}

void pool::update(float aDeltaTime)
{
	mVelocityHandling.apply(aDeltaTime);
	mBoxCollision.apply();
	mNeighborhoodCollision.apply();
	mInterParticleCollision.apply();
}

pbd::particles& pool::particles()
{
	return mParticles;
}
