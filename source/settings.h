#pragma once
#include <gvk.hpp>

namespace pbd
{
	class settings
	{
	public:
		settings() = delete;

		static void add_settings_im_gui_entries();
		static void update_settings_buffer();
		static avk::buffer& settings_buffer();

		static bool  bruteForceActive;
		static bool  rtxActive;
		static bool  greenActive;
		static bool  binarySearchActive;
		static bool  showFocusParticleNeighborhoodRange;  // sets the particle render scale of the focused particle to 1
		static bool  neighborListSorted;                  // if enabled, the list of neighbor pairs is sorted by the first of the two ids; for reduced scatter in memory access
		static float particleRenderScale;                 // how big the particles should be rendered
		static int   focusParticleId;                     // the particle for which the neighbors should be highlighted
		static int   renderedNeighborhoodMethod;          // 0: brute force, 1: RTX, 2: Green, 3: binary search

		static const std::vector<const char*> neighborhoodMethodNames;
	};
}
