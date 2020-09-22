#pragma once

#include "list_definitions.h"

namespace pbd
{
	class velocity_handling
	{
	public:
		velocity_handling& set_data(particles* aParticles);
		velocity_handling& set_acceleration(const glm::vec3& aAcceleration = glm::vec3(0));
		void apply(float aDeltaTime);

	private:
		particles* mParticles;
		glm::vec3 mAcceleration;
		float mLastDeltaTime = 1.0f;
	};
}
