#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class inter_particle_collision
	{
	public:
		inter_particle_collision& set_data(particles* aParticles, neighbors* aNeighbors);
		void apply();

	private:
		particles* mParticles;
		neighbors* mNeighbors;
	};
}
