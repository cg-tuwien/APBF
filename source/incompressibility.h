#pragma once

#include "list_definitions.h"

namespace pbd
{
	class incompressibility
	{
	public:
		incompressibility& set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * 64>* aNeighbors);
		void apply();

	private:
		fluid* mFluid;
		gpu_list<sizeof(uint32_t) * 64>* mNeighbors;
	};
}
