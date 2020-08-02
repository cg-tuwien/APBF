#pragma once

#include "list_definitions.h"

namespace pbd
{
	class velocity_handling
	{
	public:
		velocity_handling& add_particles(const particles& aParticles, const glm::vec3& aAcceleration = glm::vec3(0));
		void apply(float aDeltaTime);

	private:
		std::list<std::pair<particles, glm::vec3>> mParticles;
	};
}
