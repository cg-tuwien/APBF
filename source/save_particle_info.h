#pragma once

#include "list_definitions.h"

namespace pbd
{
	class save_particle_info
	{
	public:
		save_particle_info& set_data(fluid* aFluid, neighbors* aNeighbors);
		void apply();
		void save_as_svg(uint32_t aSvgId, const glm::vec2& aViewBoxMin, const glm::vec2& aViewBoxMax, float aRenderScale = 1.0f);

	private:
		fluid* mFluid;
		neighbors* mNeighbors;
	};
}
