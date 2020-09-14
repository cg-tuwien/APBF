#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class neighborhood_brute_force
	{
	public:
		neighborhood_brute_force& set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors);
		neighborhood_brute_force& set_range_scale(float aScale);
		void apply();

	private:
		float mRangeScale;
		particles* mParticles;
		const gpu_list<sizeof(float)>* mRange;
		gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* mNeighbors; // TODO switch to list of neighborhood pairs
	};
}
