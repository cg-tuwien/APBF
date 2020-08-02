#pragma once

#include "velocity_handling.h"

class pool
{
public:
	pool();
	void update(float aDeltaTime);

private:
	pbd::particles mFluid;
	pbd::velocity_handling mVelocityHandling;
};
