#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class spread_kernel_width
	{
	public:
		spread_kernel_width& set_data(fluid* aFluid, gpu_list<8>* aNeighbors);
		void apply();

	private:
		fluid* mFluid;
		gpu_list<8>* mNeighbors;
	};
}
