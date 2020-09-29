#pragma once
#include <gvk.hpp>

namespace pbd
{
	class settings
	{
	public:
		settings() = delete;

		static void add_apbf_settings_im_gui_entries();
		static void update_apbf_settings_buffer();
		static avk::buffer& apbf_settings_buffer();

		static int   heightKernelId;
		static int   gradientKernelId;
		static bool  matchGradientToHeightKernel;
		static bool  merge;
		static bool  split;
		static bool  baseKernelWidthOnTargetRadius;
		static bool  updateTargetRadius;
		static bool  updateBoundariness;
		static bool  groundTruthBoundaryDistance;
		static float boundarinessAdaptionSpeed;
		static float kernelWidthAdaptionSpeed;
		static float boundarinessSelfGradLengthFactor;
		static float boundarinessUnderpressureFactor;
		static float neighborBoundarinessThreshold;
		static int   color; // 0 = boundariness, 1 = boundary distance, 2 = transferring, 3 = kernel width, 4 = target radius, 5 = radius
		static int   solverIterations;
	};
}
