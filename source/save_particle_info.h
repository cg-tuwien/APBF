#pragma once

#include "list_definitions.h"

namespace pbd
{
	class save_particle_info
	{
	public:
		save_particle_info& set_data(fluid* aFluid, neighbors* aNeighbors, transfers* aTransfers);
		save_particle_info& set_boxes(const pbd::gpu_list<16>* aBoxMin, const pbd::gpu_list<16>* aBoxMax); // optional, for display in SVG
		void apply();
		void save_as_svg(uint32_t aSvgId, const glm::vec2& aViewBoxMin, const glm::vec2& aViewBoxMax, float aRenderScale = 1.0f, float aMaxExpectedBoundaryDist = 100.0f);

	private:
		std::string boxes_to_svg();

		fluid* mFluid;
		neighbors* mNeighbors;
		transfers* mTransfers;
		const pbd::gpu_list<16>* mBoxMin = nullptr;
		const pbd::gpu_list<16>* mBoxMax = nullptr;
	};
}
