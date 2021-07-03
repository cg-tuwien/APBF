#include "initialize.h"
#include <numbers>
#include "../shaders/cpu_gpu_shared_config.h"

pbd::particles pbd::initialize::add_box_shape(pbd::particles& aParticles, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, float aRadius, float aInverseDensity, const glm::vec3& aVelocity)
{
	auto& positionList     = aParticles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& velocityList     = aParticles.hidden_list().get<pbd::hidden_particles::id::velocity>();
	auto& inverseMassList  = aParticles.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList       = aParticles.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& posBackupList    = aParticles.hidden_list().get<pbd::hidden_particles::id::pos_backup>();
	auto& transferringList = aParticles.hidden_list().get<pbd::hidden_particles::id::transferring>();

	auto minPos = aMinPos;
	auto maxPos = aMaxPos;
#if DIMENSIONS < 3
	minPos.z = (aMinPos.z + aMaxPos.z) / 2 - aRadius;
	maxPos.z = (aMinPos.z + aMaxPos.z) / 2 + aRadius * 1.5f; // * 1.5 to address rounding errors
#endif
#if DIMENSIONS < 2
	minPos.y = (aMinPos.y + aMaxPos.y) / 2 - aRadius;
	maxPos.y = (aMinPos.y + aMaxPos.y) / 2 + aRadius * 1.5f; // * 1.5 to address rounding errors
#endif

	auto inverseMass = aInverseDensity / static_cast<float>(std::pow(2.0f * aRadius, DIMENSIONS));
	auto particleCount = glm::uvec3((maxPos - minPos) / (2.0f * aRadius));
	auto amountToCreate = particleCount.x * particleCount.y * particleCount.z;
	auto newParticles = aParticles.increase_length(amountToCreate);
	if (amountToCreate == 0) return newParticles;
	shader_provider::initialize_box(newParticles.index_buffer(), newParticles.length(), positionList.write().buffer(), velocityList.write().buffer(), inverseMassList.write().buffer(), radiusList.write().buffer(), transferringList.write().buffer(), minPos + aRadius, particleCount, aRadius, inverseMass, aVelocity);
	posBackupList = positionList;
	return newParticles;
}

pbd::particles pbd::initialize::add_sphere_shape(pbd::particles& aParticles, const glm::vec3& aCenter, float aShapeOuterRadius, float aShapeInnerRadius, bool aOuterRadiusMoreImportant, float aParticleRadius, float aInverseDensity, const glm::vec3& aVelocity)
{
	auto& positionList     = aParticles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& velocityList     = aParticles.hidden_list().get<pbd::hidden_particles::id::velocity>();
	auto& inverseMassList  = aParticles.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList       = aParticles.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& posBackupList    = aParticles.hidden_list().get<pbd::hidden_particles::id::pos_backup>();
	auto& transferringList = aParticles.hidden_list().get<pbd::hidden_particles::id::transferring>();
	auto  tempPositionList = gpu_list<16>();

	auto s = aOuterRadiusMoreImportant ? -1.0f : 1.0f;
	auto a = static_cast<float>(std::pow(aOuterRadiusMoreImportant ? aShapeOuterRadius : aShapeInnerRadius, DIMENSIONS));
	auto b = static_cast<float>(std::pow(aParticleRadius, DIMENSIONS) * 2 * DIMENSIONS / std::numbers::pi) * s;
	aShapeOuterRadius -= aParticleRadius;
	if (aShapeInnerRadius != 0.0f) aShapeInnerRadius += aParticleRadius;
	auto radius = aOuterRadiusMoreImportant ? aShapeOuterRadius : aShapeInnerRadius;
	auto offset = aParticleRadius * s;
	auto particleList = std::vector<glm::vec4>();

	while (aShapeInnerRadius <= radius && radius <= aShapeOuterRadius) {
		append_sphere(particleList, aCenter, radius, aParticleRadius, DIMENSIONS);
		if (aOuterRadiusMoreImportant && radius < aParticleRadius) break; // already placed center particle; stop before another one is placed at the same position
		radius = std::pow(a + b * particleList.size(), 1.0f / DIMENSIONS) + offset; // only works for DIMENSIONS = 2 or 3
	}

	auto inverseMass = aInverseDensity / static_cast<float>(std::pow(2.0f * aParticleRadius, DIMENSIONS));
	auto newParticles = aParticles.increase_length(particleList.size());
	if (particleList.size() == 0) return newParticles;

	tempPositionList.request_length(particleList.size());
	algorithms::copy_bytes(particleList.data(), tempPositionList.write().buffer(), particleList.size() * 16);
	shader_provider::initialize_sphere(newParticles.index_buffer(), newParticles.length(), tempPositionList.buffer(), positionList.write().buffer(), velocityList.write().buffer(), inverseMassList.write().buffer(), radiusList.write().buffer(), transferringList.write().buffer(), aParticleRadius, inverseMass, aVelocity);
	posBackupList = positionList;
	return newParticles;
}

void pbd::initialize::append_sphere(std::vector<glm::vec4>& aParticleList, const glm::vec3& aCenter, float aSphereRadius, float aParticleRadius, uint32_t aDimensions)
{
	if (aDimensions <= 1u) {
		auto center = glm::vec4(aCenter, 1.0f);
		auto offset = glm::vec4(aSphereRadius, 0, 0, 0);

		if (aSphereRadius < aParticleRadius) {
			aParticleList.push_back(center);
		} else {
			aParticleList.push_back(center - offset);
			aParticleList.push_back(center + offset);
		}
		return;
	}
	--aDimensions;
	append_sphere(aParticleList, aCenter, aSphereRadius, aParticleRadius, aDimensions);
	auto pCount = std::floor(aSphereRadius * static_cast<float>(std::numbers::pi) / (2.0f * aParticleRadius));
	auto dAngle = static_cast<float>(std::numbers::pi) / pCount;
	auto angle  = 0.0f;

	for (auto i = static_cast<int>(pCount) / 2; i > 0; i--) {
		angle += dAngle;
		auto layerOffset = std::sin(angle) * aSphereRadius;
		auto layerRadius = std::cos(angle) * aSphereRadius;
		auto layerCenter = aCenter;
		layerCenter[aDimensions] += layerOffset;
		append_sphere(aParticleList, layerCenter, layerRadius, aParticleRadius, aDimensions);
		layerCenter[aDimensions] -= 2 * layerOffset;
		append_sphere(aParticleList, layerCenter, layerRadius, aParticleRadius, aDimensions);
	}
}
