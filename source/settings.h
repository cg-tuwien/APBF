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
		static void nextColor();
		static void previousColor();

		static int   heightKernelId;                      // 0 = cubic, 1 = gauss, 2 = poly6, 3 = cone, 4 = quadratic spike; cubic and poly6 are only for 3D
		static int   gradientKernelId;                    // 0 = cubic, 1 = gauss, 2 = spiky, 3 = cone, 4 = quadratic spike; cubic and spiky are only for 3D
		static bool  matchGradientToHeightKernel;         // if true, a change of the height kernel also changes the gradient kernel so that they are always the same
		static bool  merge;                               // enable/disable merge; if both merge and split are disabled, all current transfers are also paused
		static bool  split;                               // enable/disable split; if both merge and split are disabled, all current transfers are also paused
		static bool  baseKernelWidthOnTargetRadius;       // not used if baseKernelWidthOnBoundaryDistance is true
		static bool  baseKernelWidthOnBoundaryDistance;   // for switching to the streamlined variant
		static bool  updateTargetRadius;                  // allows to disable the update of the target radius for debugging; TODO delete?
		static bool  updateBoundariness;                  // allows to disable the update of the boundariness  for debugging; TODO delete?
		static bool  neighborListSorted;                  // if enabled, the list of neighbor pairs is sorted by the first of the two ids; for reduced scatter in memory access
		static bool  groundTruthBoundaryDistance;         // hardcoded boundary distance; outdated; TODO delete?
		static bool  renderBoxes;                         // if the collision boxes (pool walls) should be rendered
		static bool  basicPbf;                            // if true, only run the basic PBF algorithm without split/merge etc. (will behave badly if particles already have varying sizes)
		static float boundarinessAdaptionSpeed;           // maximum step width for updating boundariness (slow down to filter short occurrences of false positives)
		static float kernelWidthAdaptionSpeed;            // maximum percent change for updating kernel width (fast kernel width changes can cause large changes in the density estimate and lead to erratic particle movement)
		static float boundarinessSelfGradLengthFactor;    // for boundariness classification: influence of the gradient length for the particle itself
		static float boundarinessUnderpressureFactor;     // for boundariness classification: influence of underpressure
		static float neighborBoundarinessThreshold;       // for filtering false positives in boundariness classification: only keep if enough neighbors are also classified as boundary particle; did not really improve the results; TODO delete?
		static float mergeDuration;                       // how long a gradual merge should take (0 for instant merge)
		static float splitDuration;                       // how long a gradual split should take (0 for instant split)
		static float smallestTargetRadius;                // smallest allowed particle radius
		static float targetRadiusOffset;                  // margin from fluid boundary which shall only contain the smallest allowed particles; not used if baseKernelWidthOnBoundaryDistance is true
		static float targetRadiusScaleFactor;             // the rate at which the target radius should increase with increasing boundary distance; if baseKernelWidthOnBoundaryDistance is true, this is actually the kernel width scale factor; TODO rename?
		static float particleRenderScale;                 // how big the particles should be rendered
		static int   particleRenderLimit;                 // for limiting the number of rendered particles (for debugging); 0 means no limit
		static int   color;                               // 0 = boundariness, 1 = boundary distance, 2 = transferring, 3 = kernel width, 4 = target radius, 5 = radius, 6 = velocity
		static int   solverIterations;                    // how often the constraint solving is repeated each timestep

		static const std::vector<const char*> colorNames;
	};
}
