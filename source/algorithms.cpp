#include "algorithms.h"

void pbd::algorithms::copy_bytes(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset, size_t aTargetOffset)
{
	if (aCopiedLength == 0) return;

	assert(aSource->meta<avk::storage_buffer_meta>().total_size() >= aSourceOffset + aCopiedLength);
	assert(aTarget->meta<avk::storage_buffer_meta>().total_size() >= aTargetOffset + aCopiedLength);

	auto copyRegion = vk::BufferCopy{}
		.setSrcOffset(aSourceOffset)
		.setDstOffset(aTargetOffset)
		.setSize(aCopiedLength);
	shader_provider::cmd_bfr()->handle().copyBuffer(aSource->buffer_handle(), aTarget->buffer_handle(), { copyRegion });
	shader_provider::sync_after_transfer();
}

size_t pbd::algorithms::sort_calculate_needed_helper_list_length(size_t aValueCount)
{
	auto subkeyLength = 4u;
	auto bucketCount = static_cast<uint32_t>(pow(2u, subkeyLength));
	auto elementCount = static_cast<uint32_t>(aValueCount);
	auto groupsize = 512u;
	auto histogramTableCount = bucketCount * ((elementCount + groupsize - 1u) / groupsize);
	return histogramTableCount + prefix_sum_calculate_needed_helper_list_length(histogramTableCount);
}

size_t pbd::algorithms::prefix_sum_calculate_needed_helper_list_length(size_t aValueCount)
{
	auto elementCount = static_cast<uint32_t>(aValueCount);
	auto groupsize = 512u;
	auto result = 0u;
	do {
		elementCount = (elementCount + groupsize - 1u) / groupsize;
		result += elementCount;
	} while (elementCount > 1);
	return result == 0u ? 0u : result + 9u;
}
void pbd::algorithms::sort(const avk::buffer& aValues, const avk::buffer& aSecondValues, const avk::buffer& aHelperList, const avk::buffer& aValueCount, const avk::buffer& aResult, const avk::buffer& aSecondResult, uint32_t aValueUpperBound)
{
	auto subkeyLength = 4u;
	auto blocksize = 512u;
	auto bucketCount = static_cast<uint32_t>(pow(2u, subkeyLength));
	auto maxValueCount = aValues->meta<avk::storage_buffer_meta>().total_size() / 4;
	auto maxHistogramTableCount = bucketCount * ((maxValueCount + blocksize - 1u) / blocksize);
	auto histogramTableOffset = aHelperList->meta<avk::storage_buffer_meta>().total_size() / 4 - maxHistogramTableCount;
	auto doPrefixSum = maxValueCount > blocksize;

	auto& histogramTable = aHelperList;

	for (auto subkeyOffset = 0u; (subkeyOffset < 32u) && (aValueUpperBound >> subkeyOffset != 0); subkeyOffset += subkeyLength)
	{
		auto lastIteration = (subkeyOffset + subkeyLength >= 32u) || (aValueUpperBound >> (subkeyOffset + subkeyLength) == 0u);
		auto inResult = doPrefixSum != lastIteration;
		shader_provider::radix_sort_apply_on_block_level(aValues, inResult ? aResult : aValues, aSecondValues, inResult ? aSecondResult : aSecondValues, histogramTable, aValueCount, histogramTableOffset, subkeyOffset, subkeyLength);

		if (doPrefixSum)
		{
			// histogram table length was written into aHelperList[0] by radix_sort_apply_on_block_level()
			prefix_sum(histogramTable, aHelperList, histogramTable, 0u, maxHistogramTableCount);
			//shader_provider::radix_sort_scattered_write();
		}
	}
}

void pbd::algorithms::prefix_sum(const avk::buffer& aValues, const avk::buffer& aHelperList, const avk::buffer& aValueCount, const avk::buffer* aResult)
{
	auto maxValueCount = aValues->meta<avk::storage_buffer_meta>().total_size() / 4;
	copy_bytes(aValueCount, aHelperList, 4);
	prefix_sum(aValues, aHelperList, aResult == nullptr ? aValues : *aResult, 0u, maxValueCount);
}

void pbd::algorithms::prefix_sum(const avk::buffer& aValues, const avk::buffer& aHelperList, const avk::buffer& aResult, uint32_t aRecursionDepth, uint32_t aMaxValueCount)
{
	// the first 10 uint of aHelperList are reserved for lengths and offsets:
	// uint 0-3 are the number of values for the recursion depths 0-3, respectively
	// uint 4-9 are the offsets of the values and the resulting group sums (uint 5 is the group sum offset for recursion 0 and the value offset for recursion 1)

	auto groupsize = 512u;
	shader_provider::prefix_sum_apply_on_block_level(aValues, aResult, aHelperList, aHelperList, aRecursionDepth);
	aMaxValueCount = (aMaxValueCount + groupsize - 1u) / groupsize;
	if (aMaxValueCount <= 1u) return;

	prefix_sum(aHelperList, aHelperList, aHelperList, aRecursionDepth + 1u, aMaxValueCount);
	shader_provider::prefix_sum_spread_from_block_level(aResult, aResult, aHelperList, aHelperList, aRecursionDepth);
}
