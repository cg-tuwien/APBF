#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class neighborhood_green
	{
	public:
		neighborhood_green& set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors);
		neighborhood_green& set_range_scale(float aScale);
		neighborhood_green& set_position_range(const glm::vec3& aMinPos, const glm::vec3& aMaxPos, uint32_t aResolutionLog2);
		void apply();

	private:
		float mRangeScale;
		particles* mParticles;
		const gpu_list<sizeof(float)>* mRange;
		gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* mNeighbors;
		glm::vec3 mMinPos, mMaxPos;
		uint32_t mResolutionLog2;
	};
}
