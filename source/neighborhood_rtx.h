#pragma once

#include "list_definitions.h"

namespace pbd
{
	class neighborhood_rtx
	{
	public:
		neighborhood_rtx();
		neighborhood_rtx& set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, neighbors* aNeighbors);
		neighborhood_rtx& set_range_scale(float aScale);
		void apply();

	private:
		void reserve_geometry_instances_buffer(size_t aSize);
		void build_acceleration_structure();

		float mRangeScale;
		particles* mParticles;
		const gpu_list<sizeof(float)>* mRange;
		neighbors* mNeighbors;
		avk::buffer mGeometryInstances;
		avk::top_level_acceleration_structure mTlas;
		avk::bottom_level_acceleration_structure mBlas;
		size_t mMaxInstanceCount;
	};
}
