#pragma once

#include "list_definitions.h"

namespace pbd
{
	class initialize
	{
	public:
		static pbd::particles add_box_shape(pbd::particles& aParticles, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, float aRadius = 1.0f, float aInverseDensity = 1.0f, const glm::vec3& aVelocity = glm::vec3(0));
		static pbd::particles add_sphere_shape(pbd::particles& aParticles, const glm::vec3& aCenter, float aShapeOuterRadius, float aShapeInnerRadius = 0.0f, bool aOuterRadiusMoreImportant = true, float aParticleRadius = 1.0f, float aInverseDensity = 1.0f, const glm::vec3& aVelocity = glm::vec3(0));

	private:
		static void append_sphere(std::vector<glm::vec4>& aParticleList, const glm::vec3& aCenter, float aSphereRadius, float aParticleRadius, uint32_t aDimensions);
	};
}
