#include "initialize.h"
#include "../shaders/cpu_gpu_shared_config.h"

pbd::particles pbd::initialize::add_box_shape(pbd::particles& aParticles, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, float aRadius, float aInverseDensity, const glm::vec3& aVelocity)
{
	auto& positionList    = aParticles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& velocityList    = aParticles.hidden_list().get<pbd::hidden_particles::id::velocity>();
	auto& inverseMassList = aParticles.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList      = aParticles.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& posBackupList   = aParticles.hidden_list().get<pbd::hidden_particles::id::pos_backup>();

	auto minPos = aMinPos;
	auto maxPos = aMaxPos;
#if DIMENSIONS < 3
	minPos.z = (aMinPos.z + aMaxPos.z) / 2 - aRadius;
	maxPos.z = (aMinPos.z + aMaxPos.z) / 2 + aRadius;
#endif
#if DIMENSIONS < 2
	minPos.y = (aMinPos.y + aMaxPos.y) / 2 - aRadius;
	maxPos.y = (aMinPos.y + aMaxPos.y) / 2 + aRadius;
#endif

	auto inverseMass = aInverseDensity / std::pow(2.0f * aRadius, 3.0f);
	auto particleCount = glm::uvec3((maxPos - minPos) / (2.0f * aRadius));
	auto amountToCreate = particleCount.x * particleCount.y * particleCount.z;
	auto newParticles = aParticles.increase_length(amountToCreate);
	shader_provider::initialize_box(aParticles.index_buffer(), aParticles.length(), positionList.write().buffer(), velocityList.write().buffer(), inverseMassList.write().buffer(), radiusList.write().buffer(), minPos + aRadius, particleCount, aRadius, inverseMass, aVelocity);
	posBackupList = positionList;
	return newParticles;
}
