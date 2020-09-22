#pragma once

#include "list_definitions.h"

namespace pbd
{
	class incompressibility
	{
	public:
		//will only write transfers into hidden list, the index list of aTransfers remains untouched
		incompressibility& set_data(fluid* aFluid, neighbors* aNeighbors);
		void apply();

	private:
		fluid* mFluid;
		neighbors* mNeighbors;
	};
}
