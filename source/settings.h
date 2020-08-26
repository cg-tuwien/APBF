#pragma once

namespace pbd
{
	class settings
	{
	public:
		settings() = delete;

		static bool merge;
		static bool split;
		static bool baseKernelWidthOnTargetRadius;
		static bool updateTargetRadius;
		static bool updateBoundariness;
		static int  color; // 0 = boundariness, 1 = boundary distance, 2 = target radius, 3 = transferring
	};
}
