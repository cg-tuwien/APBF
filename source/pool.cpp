#include "pool.h"
#include "initialize.h"

pool::pool() :
	mFluid(1024)
{
	shader_provider::start_recording();
	mFluid.request_length(1024);
	pbd::initialize::add_box_shape(mFluid, glm::vec3(0, 0, 0), glm::vec3(20, 10, 6));
	mVelocityHandling.add_particles(mFluid, glm::vec3(0, -1, 0));
	shader_provider::end_recording();
}

void pool::update(float aDeltaTime)
{
	mVelocityHandling.apply(aDeltaTime);
}
