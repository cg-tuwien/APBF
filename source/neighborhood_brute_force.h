#pragma once

#include "list_definitions.h"

namespace pbd
{
	class neighborhood_brute_force
	{
	public:
		neighborhood_brute_force& set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, neighbors* aNeighbors);
		neighborhood_brute_force& set_range_scale(float aScale);
		void apply();

	private:
		float mRangeScale;
		particles* mParticles;
		const gpu_list<sizeof(float)>* mRange;
		neighbors* mNeighbors;
	};
}
