#pragma once

#include <gvk.hpp>

class shader_provider
{
public:
	struct changing_length
	{
		avk::buffer& mOldLength;
		avk::buffer& mNewLength;
	};
	// TODO maybe add an init_all() function
	static void set_queue(avk::queue& aQueue);
	static void start_recording();
	static void end_recording();
	static avk::command_buffer& cmd_bfr();
	static void roundandround(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, uint32_t aParticleCount);
	static void append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const changing_length& aTargetListLength, const avk::buffer& aAppendingListLength, uint32_t aStride);
	static void copy_scattered_read(const avk::buffer& aSourceList, const avk::buffer& aTargetList, const avk::buffer& aEditList, const avk::buffer& aEditListLength, const avk::buffer& aNewTargetListLength, uint32_t aStride);
	static void write_increasing_sequence(const avk::buffer& aTargetList, const avk::buffer& aNewTargetListLength, const changing_length& aSequenceMinValue, uint32_t aValueUpperBound, uint32_t aSequenceLength);
	static void prefix_sum_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aOutGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void prefix_sum_spread_from_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void radix_sort_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aOutHistogramTable, const avk::buffer& aBufferLength, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);
	static void radix_sort_scattered_write(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aInHistogramTable, const avk::buffer& aBufferLength, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);

	static void sync_after_compute();
	static void sync_after_transfer();
private:
	static avk::descriptor_cache& descriptor_cache();
	static const avk::buffer& workgroup_count_buffer();
	static void dispatch(uint32_t aX = 1u, uint32_t aY = 1u, uint32_t aZ = 1u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aOffset = 0u, uint32_t aScalingFactor = 1u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void dispatch_indirect();

	static avk::command_buffer mCmdBfr;
	static avk::queue* mQueue;
};
