#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class neighborhood_binary_search
	{
	public:
		neighborhood_binary_search& set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, pbd::neighbors* aNeighbors);
		neighborhood_binary_search& set_range_scale(float aScale);
		void apply();

	private:
		float mRangeScale;
		particles* mParticles;
		const gpu_list<sizeof(float)>* mRange;
		pbd::neighbors* mNeighbors;
	};
}
