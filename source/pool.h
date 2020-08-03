#pragma once

#include "velocity_handling.h"
#include "box_collision.h"

class pool
{
public:
	pool(const glm::vec3& aMin, const glm::vec3& aMax);
	void update(float aDeltaTime);
	pbd::particles& particles();

private:
	pbd::particles mFluid;
	pbd::velocity_handling mVelocityHandling;
	pbd::box_collision mBoxCollision;
};
