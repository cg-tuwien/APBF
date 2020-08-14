#pragma once

#include "shader_provider.h"

namespace pbd
{
	class algorithms
	{
	public:
		algorithms() = delete;

		static void copy_bytes(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset = 0, size_t aTargetOffset = 0);
		static void copy_bytes(const void* aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset = 0, size_t aTargetOffset = 0);
		static size_t sort_calculate_needed_helper_list_length(size_t aMaxValueCount);
		static size_t prefix_sum_calculate_needed_helper_list_length(size_t aMaxValueCount);
		static void sort(const avk::buffer& aValues, const avk::buffer& aSecondValues, const avk::buffer& aHelperList, const avk::buffer& aValueCount, size_t aMaxValueCount, const avk::buffer& aResult, const avk::buffer& aSecondResult, uint32_t aValueUpperBound = MAXUINT32);
		static void prefix_sum(const avk::buffer& aValues, const avk::buffer& aHelperList, const avk::buffer& aValueCount, size_t aMaxValueCount, const avk::buffer* aResult = nullptr);
	
	private:
		// aHelperList[aHelperList.length - 10 + aRecursionDepth    ] has to be set to the length of aValues
		// aHelperList[aHelperList.length - 10 + aRecursionDepth + 4] has to be set to the offset of aValues
		// aHelperList[aHelperList.length - 10 + aRecursionDepth + 5] has to be set to the offset of aResult
		static void prefix_sum_recursive(const avk::buffer& aValues, const avk::buffer& aHelperList, const avk::buffer& aResult, uint32_t aRecursionDepth, uint32_t aMaxValueCount = MAXUINT32);
		static const avk::buffer& zero();
	};
}
