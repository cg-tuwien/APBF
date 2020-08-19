#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class update_transfers
	{
	public:
		//will only write transfers into hidden list, the index list of aTransfers remains untouched
		update_transfers& set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors, transfers* aTransfers);
		void apply();

	private:
		fluid* mFluid;
		gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* mNeighbors;
		transfers* mTransfers;
	};
}