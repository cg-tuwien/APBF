#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class incompressibility
	{
	public:
		//will only write transfers into hidden list, the index list of aTransfers remains untouched
		incompressibility& set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors, gpu_list<8>* aNeighbors2);
		void apply();

	private:
		fluid* mFluid;
		gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* mNeighbors;
		gpu_list<8>* mNeighbors2;
	};
}
