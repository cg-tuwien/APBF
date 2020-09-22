#pragma once

#include "list_definitions.h"

namespace pbd
{
	class spread_kernel_width
	{
	public:
		spread_kernel_width& set_data(fluid* aFluid, neighbors* aNeighbors);
		void apply();

	private:
		fluid* mFluid;
		neighbors* mNeighbors;
	};
}
