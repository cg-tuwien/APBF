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

void pbd::algorithms::copy_bytes(const void* aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset, size_t aTargetOffset)
{
	if (aCopiedLength == 0) return;

	aSource = static_cast<const void*>(static_cast<const char8_t*>(aSource) + aSourceOffset);
	assert(aTarget->meta<avk::storage_buffer_meta>().total_size() >= aTargetOffset + aCopiedLength);

	auto stagingBuffer = gvk::context().create_buffer(
		avk::memory_usage::host_coherent, vk::BufferUsageFlagBits::eTransferSrc,
		avk::storage_buffer_meta::create_from_size(aCopiedLength)
	);
	stagingBuffer->fill(aSource, 0, avk::sync::wait_idle());
	shader_provider::sync_after_transfer();
	copy_bytes(stagingBuffer, aTarget, aCopiedLength, 0, aTargetOffset);
	shader_provider::cmd_bfr()->set_custom_deleter([
		lOwnedStagingBuffer{ std::move(stagingBuffer) }
	]() { /* Nothing to do here, the buffers' destructors will do the cleanup, the lambda is just storing it. */ });
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
	return result == 0u ? 0u : result + 10u;
}
void pbd::algorithms::sort(const avk::buffer& aValues, const avk::buffer& aSecondValues, const avk::buffer& aHelperList, const avk::buffer& aValueCount, const avk::buffer& aResult, const avk::buffer& aSecondResult, uint32_t aValueUpperBound)
{
	auto subkeyLength = 4u;
	auto blocksize = 512u;
	auto bucketCount = static_cast<uint32_t>(pow(2u, subkeyLength));
	auto maxValueCount = static_cast<uint32_t>(aValues->meta<avk::storage_buffer_meta>().total_size()) / 4u;
	auto maxHistogramTableCount = bucketCount * ((maxValueCount + blocksize - 1u) / blocksize);
	auto lengthsAndOffsetsOffset = static_cast<uint32_t>(aHelperList->meta<avk::storage_buffer_meta>().total_size()) / 4u - 10u;
	auto doPrefixSum = maxValueCount > blocksize;

	auto& histogramTable = aHelperList;

	for (auto subkeyOffset = 0u; (subkeyOffset < 32u) && (aValueUpperBound >> subkeyOffset != 0); subkeyOffset += subkeyLength)
	{
		auto lastIteration = (subkeyOffset + subkeyLength >= 32u) || (aValueUpperBound >> (subkeyOffset + subkeyLength) == 0u);
		auto inResult = doPrefixSum != lastIteration;
		shader_provider::radix_sort_apply_on_block_level(aValues, inResult ? aResult : aValues, aSecondValues, inResult ? aSecondResult : aSecondValues, histogramTable, aValueCount, aHelperList, lengthsAndOffsetsOffset, subkeyOffset, subkeyLength);

		if (doPrefixSum)
		{
			// histogram table length was written into aHelperList[0] and aHelperList[5] by radix_sort_apply_on_block_level(), as well as 0 into aHelperList[4]
			prefix_sum(histogramTable, aHelperList, histogramTable, 0u, maxHistogramTableCount);
			if (lastIteration) {
				shader_provider::radix_sort_scattered_write(aValues, aResult, aSecondValues, aSecondResult, histogramTable, aValueCount, subkeyOffset, subkeyLength);
			} else {
				shader_provider::radix_sort_scattered_write(aResult, aValues, aSecondResult, aSecondValues, histogramTable, aValueCount, subkeyOffset, subkeyLength);
			}
		}
	}
}

void pbd::algorithms::prefix_sum(const avk::buffer& aValues, const avk::buffer& aHelperList, const avk::buffer& aValueCount, const avk::buffer* aResult)
{
	auto lengthsAndOffsetsOffset = static_cast<uint32_t>(aHelperList->meta<avk::storage_buffer_meta>().total_size()) / 4u - 10u;
	auto maxValueCount = static_cast<uint32_t>(aValues->meta<avk::storage_buffer_meta>().total_size()) / 4u;
	copy_bytes(aValueCount, aHelperList, 4, 0, lengthsAndOffsetsOffset * 4);
	copy_bytes(zero(), aHelperList, 4, 0, (lengthsAndOffsetsOffset + 4) * 4);
	copy_bytes(zero(), aHelperList, 4, 0, (lengthsAndOffsetsOffset + 5) * 4);
	prefix_sum(aValues, aHelperList, aResult == nullptr ? aValues : *aResult, 0u, maxValueCount);
}

void pbd::algorithms::prefix_sum(const avk::buffer& aValues, const avk::buffer& aHelperList, const avk::buffer& aResult, uint32_t aRecursionDepth, uint32_t aMaxValueCount)
{
	// the last 10 uint of aHelperList are reserved for lengths and offsets:
	// uint 0-3 are the number of values for the recursion depths 0-3, respectively
	// uint 4-9 are the offsets of the values and the resulting group sums (uint 5 is the group sum offset for recursion 0 and the value offset for recursion 1)
	
	auto lengthsAndOffsetsOffset = static_cast<uint32_t>(aHelperList->meta<avk::storage_buffer_meta>().total_size()) / 4u - 10u;

	auto groupsize = 512u;
	shader_provider::prefix_sum_apply_on_block_level(aValues, aResult, aHelperList, aHelperList, lengthsAndOffsetsOffset, aRecursionDepth);
	aMaxValueCount = (aMaxValueCount + groupsize - 1u) / groupsize;
	if (aMaxValueCount <= 1u) return;

	prefix_sum(aHelperList, aHelperList, aHelperList, aRecursionDepth + 1u, aMaxValueCount);
	shader_provider::prefix_sum_spread_from_block_level(aResult, aResult, aHelperList, aHelperList, lengthsAndOffsetsOffset, aRecursionDepth);
}

const avk::buffer& pbd::algorithms::zero()
{
	static auto value = 0u;
	static avk::buffer buffer = gvk::context().create_buffer(
		avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
		avk::storage_buffer_meta::create_from_size(4)
	);
	if (value == 0u) {
		buffer->fill(&value, 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr(), {}, {}));
		shader_provider::sync_after_transfer();
		value = 1u;
	}
	return buffer;
}
