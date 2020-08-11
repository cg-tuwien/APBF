#pragma once

#include "list_definitions.h"

namespace pbd
{
	class box_collision
	{
	public:
		using boxes = pbd::indexed_list<pbd::gpu_list<32ui64>>;

		box_collision();
		box_collision& add_particles(const particles& aParticles);
		boxes add_box(const glm::vec3& aMin, const glm::vec3& aMax);
		void delete_boxes(boxes& aBox);
		void apply();

	private:
		particles mParticles;
		boxes mBoxes;
	};
}
