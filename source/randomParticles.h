#pragma once
#include <gvk.hpp>
#include "list_definitions.h"

class randomParticles
{
public:
	void init(unsigned int aParticleCount, const glm::vec3& aMin, const glm::vec3& aMax);
	pbd::particles& particles();
private:
	pbd::particles mParticles;
};
