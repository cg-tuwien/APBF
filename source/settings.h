#pragma once
#include <gvk.hpp>
#include "../shaders/cpu_gpu_shared_config.h"

namespace pbd
{
	class settings
	{
	public:
		settings() = delete;

		static void add_apbf_settings_im_gui_entries();
		static void update_apbf_settings_buffer();
		static avk::buffer& apbf_settings_buffer();

		static int  heightKernelId;
		static int  gradientKernelId;
		static bool matchGradientToHeightKernel;
		static bool merge;
		static bool split;
		static bool baseKernelWidthOnTargetRadius;
		static bool updateTargetRadius;
		static bool updateBoundariness;
		static bool groundTruthBoundaryDistance;
		static float boundarinessAdaptionSpeed;
		static float kernelWidthAdaptionSpeed;
		static int  color; // 0 = boundariness, 1 = boundary distance, 2 = target radius, 3 = transferring
	};
}
