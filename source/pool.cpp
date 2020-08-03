#include "pool.h"
#include "initialize.h"

pool::pool(const glm::vec3& aMin, const glm::vec3& aMax) :
	mFluid(1024)
{
	shader_provider::start_recording();
	mFluid.request_length(1024);
	pbd::initialize::add_box_shape(mFluid, aMin + glm::vec3(2, 20, 2), aMax - glm::vec3(2, 2, 2));
	mVelocityHandling.add_particles(mFluid, glm::vec3(0, -10, 0));
	mBoxCollision.add_particles(mFluid);
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z));
	shader_provider::end_recording();
}

void pool::update(float aDeltaTime)
{
	mVelocityHandling.apply(aDeltaTime);
	mBoxCollision.apply();
}

pbd::particles& pool::particles()
{
	return mFluid;
}
