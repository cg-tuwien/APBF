#pragma once

#include "list_definitions.h"
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class particle_transfer
	{
	public:
		// only the hidden list of aTransfers will be used, not its index list!
		particle_transfer& set_data(fluid* aFluid, transfers* aTransfers);
		void apply(float aDeltaTime);

	private:
		fluid* mFluid;
		transfers* mTransfers;
	};
}
