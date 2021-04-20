#pragma once

#include "list_definitions.h"

namespace pbd
{
	class glue
	{
	private:
		enum class glue_enum { particle_0, particle_1, distance };
	public:
		using glue_list = uninterleaved_list<glue_enum, particles, particles, gpu_list<4>>;

		glue& set_data(particles* aParticles, neighbors* aNeighbors, glue_list* aGlue);
		glue& set_stability(float aStability);   // How fast two glued particles are pulled back together
		glue& set_elasticity(float aElasticity); // How far apart two glued particles can go before the glue breaks
		void apply();                            // Let glue pull particles together + break overstretched glue
		void applyNewGlue();                     // Remove old glue, then glue together all particles that are close together

	private:
		particles* mParticles;
		neighbors* mNeighbors;
		glue_list* mGlue;
		
		float mStability  = 1.0f;
		float mElasticity = 0.0f;
	};
}
