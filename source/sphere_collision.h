#pragma once

#include "list_definitions.h"

namespace pbd
{
	class sphere_collision
	{
	public:
		sphere_collision& set_data(particles* aParticles);
		sphere_collision& set_sphere(const glm::vec3& aCenter, float aRadius, bool aHollow = false);
		void apply();

	private:
		particles* mParticles;
		glm::vec3 mCenter;
		float mRadius;
		bool mHollow;
	};
}
