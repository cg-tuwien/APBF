#include "sphere_collision.h"
#include "measurements.h"

pbd::sphere_collision& pbd::sphere_collision::set_data(particles* aParticles)
{
	mParticles = aParticles;
	return *this;
}

pbd::sphere_collision& pbd::sphere_collision::set_sphere(const glm::vec3& aCenter, float aRadius, bool aHollow)
{
	mCenter = aCenter;
	mRadius = aRadius;
	mHollow = aHollow;
	return *this;
}

void pbd::sphere_collision::apply()
{
	if (mParticles->empty()) return;

	measurements::debug_label_start("Sphere collision constraints", glm::vec4(0, 0.5f, 1, 1));

	auto& positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& radiusList   = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();

	shader_provider::sphere_collision(mParticles->index_buffer(), positionList.write().buffer(), radiusList.buffer(), mParticles->length(), mCenter, mRadius, mHollow);

	measurements::debug_label_end();
}
