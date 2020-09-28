#pragma once

#include "list_definitions.h"

namespace pbd
{
	class box_collision
	{
	public:
		box_collision& set_data(particles* aParticles, pbd::gpu_list<16>* aBoxMin, pbd::gpu_list<16>* aBoxMax);
		void apply();

	private:
		particles* mParticles;
		pbd::gpu_list<16>* mBoxMin;
		pbd::gpu_list<16>* mBoxMax;
	};
}
