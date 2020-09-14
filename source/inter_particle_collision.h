#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class inter_particle_collision
	{
	public:
		inter_particle_collision& set_data(particles* aParticles, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors);
		void apply();

	private:
		particles* mParticles;
		gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* mNeighbors; // TODO switch to list of neighborhood pairs
	};
}
