#pragma once

#include <gvk.hpp>

class shader_provider
{
public:
	// TODO maybe add an init_all() function
	static void set_queue(avk::queue& aQueue);
	static void set_updater(gvk::updater* aUpdater);
	static void start_recording();
	static void end_recording(std::optional<avk::resource_reference<avk::semaphore_t>> aWaitSemaphore = {});
	static bool is_recording();
	static avk::command_buffer& cmd_bfr();
	static const avk::buffer& append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const avk::buffer& aTargetListLength, const avk::buffer& aAppendingListLength, uint32_t aStride);
	static void copy_scattered_read(const avk::buffer& aSourceList, const avk::buffer& aTargetList, const avk::buffer& aEditList, const avk::buffer& aEditListLength, uint32_t aStride);
	static void copy_with_differing_stride(const avk::buffer& aSourceList, const avk::buffer& aTargetList, const avk::buffer& aCopyLength, uint32_t aSourceStride, uint32_t aTargetStride);
	static void scattered_write(const avk::buffer& aInIndexList, const avk::buffer& aOutBuffer, const avk::buffer& aInIndexListLength, uint32_t aValue);
	static void write_sequence(const avk::buffer& aOutBuffer, const avk::buffer& aInBufferLength, uint32_t aStartValue, uint32_t aSequenceValueStep);
	static void write_sequence_float(const avk::buffer& aOutBuffer, const avk::buffer& aInBufferLength, float aStartValue, float aSequenceValueStep);
	static void write_increasing_sequence_from_to(const avk::buffer& aOutBuffer, const avk::buffer& aOutBufferLength, const avk::buffer& aInFrom, const avk::buffer& aInTo, const avk::buffer& aInBufferMaxLength);
	static const avk::buffer& write_increasing_sequence(const avk::buffer& aTargetList, const avk::buffer& aNewTargetListLength, const avk::buffer& aSequenceMinValue, uint32_t aValueUpperBound, uint32_t aSequenceLength);
	static void find_value_ranges(const avk::buffer& aInIndexBuffer, const avk::buffer& aInBuffer, const avk::buffer& aOutRangeStart, const avk::buffer& aOutRangeEnd, const avk::buffer& aInBufferLength);
	static void find_value_changes(const avk::buffer& aInBuffer, const avk::buffer& aOutChange, const avk::buffer& aInBufferLength, const avk::buffer& aOutChangeLength);
	static void atomic_swap(const avk::buffer& aInIndexList, const avk::buffer& aInOutSwapA, const avk::buffer& aInOutSwapB, const avk::buffer& aInIndexListLength);
	static void generate_new_index_and_edit_list(const avk::buffer& aInEditList, const avk::buffer& aInHiddenIdToIdxListId, const avk::buffer& aInIndexListEqualities, const avk::buffer& aOutNewIndexList, const avk::buffer& aOutNewEditList, const avk::buffer& aInEditListLength, const avk::buffer& aInOutNewLength, uint32_t aMaxNewLength);
	static void neighbor_list_to_particle_mask(const avk::buffer& aInIndexList, const avk::buffer& aInNeighbors, const avk::buffer& aOutMask, const avk::buffer& aInNeighborListLength, glm::uint aFocusParticleId);

	static void prefix_sum_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aOutGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void prefix_sum_spread_from_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void radix_sort_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aOutHistogramTable, const avk::buffer& aBufferLength, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);
	static void radix_sort_scattered_write(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aInHistogramTable, const avk::buffer& aBufferLength, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);


	static void linked_list_to_neighbor_list(const avk::buffer& aInLinkedList, const avk::buffer& aInPrefixSum, const avk::buffer& aOutNeighborList, const avk::buffer& aInParticleCount, const avk::buffer& aOutNeighborCount);
	static void neighborhood_brute_force(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, const avk::buffer& aInOutNeighborsLength, float aRangeScale);
	static void neighborhood_green(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aInCellStart, const avk::buffer& aInCellEnd, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, const avk::buffer& aInOutNeighborsLength, float aRangeScale, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, uint32_t aResolutionLog2);
	static void neighborhood_binary_search(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInPosCode0, const avk::buffer& aInPosCode1, const avk::buffer& aInPosCode2, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, const avk::buffer& aInOutNeighborsLength, float aRangeScale);
	static void neighborhood_rtx(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, const avk::buffer& aInOutNeighborsLength, const avk::top_level_acceleration_structure& aInTlas, float aRangeScale);
	static void generate_acceleration_structure_instances(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutInstances, const avk::buffer& aInIndexListLength, uint64_t aBlasReference, float aRangeScale, uint32_t aMaxInstanceCount);
	static void calculate_position_code(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aOutCode, const avk::buffer& aInPositionLength, uint32_t aCodeSection);
	static void calculate_position_hash(const avk::buffer& aInPosition, const avk::buffer& aOutHash, const avk::buffer& aInPositionLength, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, uint32_t aResolutionLog2);

	static void render_particles(const avk::buffer& aInCameraDataBuffer, const avk::buffer& aInVertexBuffer, const avk::buffer& aInIndexBuffer, const avk::buffer& aInPosition, const avk::buffer& aInRadius, const avk::buffer& aInFloatForColor, const avk::buffer& aInParticleCount, avk::image_view& aOutNormal, avk::image_view& aOutDepth, avk::image_view& aOutColor, uint32_t aIndexCount, const glm::vec3& aColor1, const glm::vec3& aColor2, float aColor1Float, float aColor2Float);

	static void sync_after_compute();
	static void sync_after_transfer();
	static void sync_after_draw();
private:
	static avk::compute_pipeline&& with_hot_reload(avk::compute_pipeline&& aPipeline);
	static avk::graphics_pipeline&& with_hot_reload(avk::graphics_pipeline&& aPipeline);
	static avk::descriptor_cache& descriptor_cache();
	static const avk::buffer& draw_indexed_indirect_command_buffer();
	static const avk::buffer& workgroup_count_buffer();
	static const avk::buffer& length_result_buffer();
	static void dispatch(uint32_t aX = 1u, uint32_t aY = 1u, uint32_t aZ = 1u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void prepare_draw_indexed_indirect(const avk::buffer& aInstanceCount, uint32_t aIndexCount);
	static void prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aOffset = 0u, uint32_t aScalingFactor = 1u, uint32_t aMinThreadCount = 0u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void dispatch_indirect();

	static avk::command_buffer mCmdBfr;
	static avk::queue* mQueue;
	static gvk::updater* mUpdater;
	static bool mIsRecording;
};
