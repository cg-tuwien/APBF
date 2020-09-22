#pragma once

#include "list_definitions.h"

namespace pbd
{
	class initialize
	{
	public:
		static pbd::particles add_box_shape(pbd::particles& aParticles, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, float aRadius = 1.0f, float aInverseDensity = 1.0f, const glm::vec3& aVelocity = glm::vec3(0));
	};
}
