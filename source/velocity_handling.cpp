#include "velocity_handling.h"

pbd::velocity_handling& pbd::velocity_handling::add_particles(const particles& aParticles, const glm::vec3& aAcceleration)
{
	mParticles.push_back(std::make_pair(aParticles, aAcceleration));
	return *this;
}

void pbd::velocity_handling::apply(float aDeltaTime)
{
	for (auto& pair : mParticles) {
		auto&  velocityList = pair.first.hidden_list().get<pbd::hidden_particles::id::velocity>();
		auto&  positionList = pair.first.hidden_list().get<pbd::hidden_particles::id::position>();
		auto& posBackupList = pair.first.hidden_list().get<pbd::hidden_particles::id::pos_backup>();

		if (aDeltaTime == 0.0f) {
			posBackupList = positionList;
			return;
		}

		shader_provider::infer_velocity(pair.first.index_buffer(), posBackupList.buffer(), positionList.buffer(), velocityList.write().buffer(), pair.first.length(), aDeltaTime);
		posBackupList = positionList;
		shader_provider::apply_acceleration(pair.first.index_buffer(), velocityList.write().buffer(), pair.first.length(), pair.second * aDeltaTime);
		shader_provider::apply_velocity(pair.first.index_buffer(), velocityList.buffer(), positionList.write().buffer(), pair.first.length(), aDeltaTime);
	}
}
