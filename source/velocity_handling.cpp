#include "velocity_handling.h"

pbd::velocity_handling& pbd::velocity_handling::set_data(particles* aParticles)
{
	mParticles = aParticles;
	return *this;
}

pbd::velocity_handling& pbd::velocity_handling::set_acceleration(const glm::vec3& aAcceleration)
{
	mAcceleration = aAcceleration;
	return *this;
}

void pbd::velocity_handling::apply(float aDeltaTime)
{
	auto&  velocityList = mParticles->hidden_list().get<pbd::hidden_particles::id::velocity>();
	auto&  positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& posBackupList = mParticles->hidden_list().get<pbd::hidden_particles::id::pos_backup>();

	if (aDeltaTime == 0.0f) {
		posBackupList = positionList;
		return;
	}

	shader_provider::infer_velocity(mParticles->index_buffer(), posBackupList.buffer(), positionList.buffer(), velocityList.write().buffer(), mParticles->length(), mLastDeltaTime);
	posBackupList = positionList;
	shader_provider::apply_acceleration(mParticles->index_buffer(), velocityList.write().buffer(), mParticles->length(), mAcceleration * aDeltaTime);
	shader_provider::apply_velocity(mParticles->index_buffer(), velocityList.buffer(), positionList.write().buffer(), mParticles->length(), aDeltaTime);
	mLastDeltaTime = aDeltaTime;
}
