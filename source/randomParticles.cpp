#include "randomParticles.h"
#include <random>

void randomParticles::init(unsigned int aParticleCount, const glm::vec3& aMin, const glm::vec3& aMax)
{
	auto positions     = std::vector<glm::vec4>();
	auto radii         = std::vector<    float>();
	auto generator     = std::default_random_engine();
	auto distributionX = std::uniform_real_distribution<float>(aMin.x, aMax.x);
	auto distributionY = std::uniform_real_distribution<float>(aMin.y, aMax.y);
	auto distributionZ = std::uniform_real_distribution<float>(aMin.z, aMax.z);
	auto distributionR = std::uniform_real_distribution<float>(  1.0f,  10.0f);

	generator.seed(10);
	positions.reserve(aParticleCount);
	radii    .reserve(aParticleCount);

	for (auto i = 0u; i < aParticleCount; i++) {
		positions.push_back(glm::vec4(distributionX(generator), distributionY(generator), distributionZ(generator), 1.0f));
		radii    .push_back(distributionR(generator));
	}
	
	mParticles.hidden_list().set_length(aParticleCount);
	auto& gpuPositions = mParticles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& gpuRadii     = mParticles.hidden_list().get<pbd::hidden_particles::id::radius  >();
	pbd::algorithms::copy_bytes(positions.data(), gpuPositions.write().buffer(), aParticleCount * sizeof(glm::vec4));
	pbd::algorithms::copy_bytes(radii    .data(), gpuRadii    .write().buffer(), aParticleCount * sizeof(    float));
}

pbd::particles& randomParticles::particles()
{
	return mParticles;
}
