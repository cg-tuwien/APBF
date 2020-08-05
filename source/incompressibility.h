#pragma once

#include "list_definitions.h"

namespace pbd
{
	class incompressibility
	{
	public:
		incompressibility& set_data(particles* aParticles, gpu_list<sizeof(uint32_t) * 64>* aNeighbors);
		void apply();

	private:
		particles* mParticles;
		gpu_list<sizeof(uint32_t) * 64>* mNeighbors;
	};
}
