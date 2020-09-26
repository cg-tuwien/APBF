#pragma once

#include <gvk.hpp>

class shader_provider
{
public:
	// TODO maybe add an init_all() function
	static void set_queue(avk::queue& aQueue);
	static void start_recording();
	static void end_recording();
	static avk::command_buffer& cmd_bfr();
	static void roundandround(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, const avk::top_level_acceleration_structure_t& aTlas, uint32_t aParticleCount);
	static void mask_neighborhood(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, const avk::top_level_acceleration_structure_t& aTlas, uint32_t aParticleCount);
	static const avk::buffer& append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const avk::buffer& aTargetListLength, const avk::buffer& aAppendingListLength, uint32_t aStride);
	static void copy_scattered_read(const avk::buffer& aSourceList, const avk::buffer& aTargetList, const avk::buffer& aEditList, const avk::buffer& aEditListLength, uint32_t aStride);
	static void scattered_write(const avk::buffer& aInIndexList, const avk::buffer& aOutBuffer, const avk::buffer& aInIndexListLength, uint32_t aValue);
	static void write_sequence(const avk::buffer& aOutBuffer, const avk::buffer& aInBufferLength, uint32_t aStartValue, uint32_t aSequenceValueStep);
	static void write_sequence_float(const avk::buffer& aOutBuffer, const avk::buffer& aInBufferLength, float aStartValue, float aSequenceValueStep);
	static void write_increasing_sequence_from_to(const avk::buffer& aOutBuffer, const avk::buffer& aOutBufferLength, const avk::buffer& aInFrom, const avk::buffer& aInTo, const avk::buffer& aInBufferMaxLength);
	static const avk::buffer& write_increasing_sequence(const avk::buffer& aTargetList, const avk::buffer& aNewTargetListLength, const avk::buffer& aSequenceMinValue, uint32_t aValueUpperBound, uint32_t aSequenceLength);
	static void find_value_ranges(const avk::buffer& aInBuffer, const avk::buffer& aOutRangeStart, const avk::buffer& aOutRangeEnd, const avk::buffer& aInBufferLength);
	static void find_value_changes(const avk::buffer& aInBuffer, const avk::buffer& aOutChange, const avk::buffer& aInBufferLength, const avk::buffer& aOutChangeLength);
	static void indexed_subtract(const avk::buffer& aInIndexList, const avk::buffer& aInMinuend, const avk::buffer& aInSubtrahend, const avk::buffer& aOutDifference, const avk::buffer& aInIndexListLength);
	static void uint_to_float(const avk::buffer& aInUintBuffer, const avk::buffer& aOutFloatBuffer, const avk::buffer& aInUintBufferLength, float aFactor);
	static void uint_to_float_but_gradual(const avk::buffer& aInUintBuffer, const avk::buffer& aOutFloatBuffer, const avk::buffer& aInUintBufferLength, float aFactor, float aMaxAdationStep);
	static void generate_new_index_list(const avk::buffer& aInRangeEnd, const avk::buffer& aOutBuffer, const avk::buffer& aInRangeEndLength, const avk::buffer& aOutBufferLength);
	static void generate_new_edit_list(const avk::buffer& aInIndexList, const avk::buffer& aInEditList, const avk::buffer& aInRangeStart, const avk::buffer& aInTargetIndex, const avk::buffer& aOutBuffer, const avk::buffer& aInIndexListLength);
	static void atomic_swap(const avk::buffer& aInIndexList, const avk::buffer& aInOutSwapA, const avk::buffer& aInOutSwapB, const avk::buffer& aInIndexListLength);
	static void generate_new_index_and_edit_list(const avk::buffer& aInEditList, const avk::buffer& aInHiddenIdToIdxListId, const avk::buffer& aInIndexListEqualities, const avk::buffer& aOutNewIndexList, const avk::buffer& aOutNewEditList, const avk::buffer& aInEditListLength, const avk::buffer& aInOutNewLength, uint32_t aMaxNewLength);
	static void prefix_sum_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aOutGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void prefix_sum_spread_from_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void radix_sort_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aOutHistogramTable, const avk::buffer& aBufferLength, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);
	static void radix_sort_scattered_write(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aInHistogramTable, const avk::buffer& aBufferLength, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);

	static void initialize_box(const avk::buffer& aInIndexList, const avk::buffer& aInIndexListLength, const avk::buffer& aOutPosition, const avk::buffer& aOutVelocity, const avk::buffer& aOutInverseMass, const avk::buffer& aOutRadius, const glm::vec3& aMinPos, const glm::uvec3& aParticleCount, float aRadius, float aInverseMass, const glm::vec3& aVelocity);

	static void add_box(const avk::buffer& aInIndexList, const avk::buffer& aOutBoxes, const glm::vec4& aMin, const glm::vec4& aMax);
	static void box_collision(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInBoxes, const avk::buffer& aInIndexListLength, const avk::buffer& aInBoxesLength);
	static void neighborhood_brute_force(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, float aRangeScale);
	static void neighborhood_green(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aInCellStart, const avk::buffer& aInCellEnd, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, float aRangeScale, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, uint32_t aResolutionLog2);
	static void neighborhood_rtx(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, const avk::top_level_acceleration_structure_t& aInTlas, float aRangeScale);
	static void neighborhood_rtx_2(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, const avk::buffer& aInOutNeighborsLength, const avk::top_level_acceleration_structure_t& aInTlas, float aRangeScale);
	static void generate_acceleration_structure_instances(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutInstances, const avk::buffer& aInIndexListLength, uint64_t aBlasReference, float aRangeScale, uint32_t aMaxInstanceCount);
	static void calculate_position_hash(const avk::buffer& aInPosition, const avk::buffer& aOutHash, const avk::buffer& aInPositionLength, const glm::vec3& aMinPos, const glm::vec3& aMaxPos, uint32_t aResolutionLog2);
	static void kernel_width(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRadius, const avk::buffer& aInTargetRadius, const avk::buffer& aInOldKernelWidth, const avk::buffer& aInOutKernelWidth, const avk::buffer& aInNeighbors, const avk::buffer& aOutNeighbors, const avk::buffer& aInNeighborsLength, const avk::buffer& aInOutNeighborsLength);
	static void inter_particle_collision(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInNeighbors, const avk::buffer& aInIndexListLength);
	static void incompressibility_0(const avk::buffer& aInIndexList, const avk::buffer& aInInverseMass, const avk::buffer& aInKernelWidth, const avk::buffer& aOutIncompData, const avk::buffer& aInIndexListLength);
	static void incompressibility_1(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInKernelWidth, const avk::buffer& aInNeighbors, const avk::buffer& aInOutIncompData, const avk::buffer& aOutScaledGradient, const avk::buffer& aInNeighborsLength);
	static void incompressibility_2(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInIncompData, const avk::buffer& aOutBoundariness, const avk::buffer& aOutLambda, const avk::buffer& aInIndexListLength);
	static void incompressibility_3(const avk::buffer& aInIndexList, const avk::buffer& aInInverseMass, const avk::buffer& aInNeighbors, const avk::buffer& aInScaledGradient, const avk::buffer& aInLambda, const avk::buffer& aInOutPosition, const avk::buffer& aInNeighborsLength);
	static void find_split_and_merge_1(const avk::buffer& aInNeighbors, const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInOldBoundaryDist, const avk::buffer& aInOutBoundaryDist, const avk::buffer& aInOutMinNeighborSqDist, const avk::buffer& aInNeighborsLength);
	static void find_split_and_merge_2(const avk::buffer& aInNeighbors, const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInMinNeighborSqDist, const avk::buffer& aOutNearestNeighbor, const avk::buffer& aInNeighborsLength);
	static void find_split_and_merge_3(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRadius, const avk::buffer& aInBoundariness, const avk::buffer& aInOutBoundaryDist, const avk::buffer& aOutTargetRadius, const avk::buffer& aInNearestNeighbor, const avk::buffer& aOutTransferSource, const avk::buffer& aOutTransferTarget, const avk::buffer& aOutTransferTimeLeft, const avk::buffer& aInOutTransferring, const avk::buffer& aOutSplit, const avk::buffer& aInIndexListLength, const avk::buffer& aInOutTransferLength, const avk::buffer& aInOutSplitLength, uint32_t aMaxTransferLength, uint32_t aMaxSplitLength);
	static const avk::buffer& remove_impossible_splits(const avk::buffer& aInSplit, const avk::buffer& aOutTransferring, const avk::buffer& aInTransferLength, const avk::buffer& aInParticleLength, const avk::buffer& aInSplitLength, uint32_t aMaxTransferLength, uint32_t aMaxParticleLength);
	static void initialize_split_particles(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aOutInverseMass, const avk::buffer& aInOutRadius, const avk::buffer& aInIndexListLength);
	static void particle_transfer(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInOutRadius, const avk::buffer& aInOutInverseMass, const avk::buffer& aInOutVelocity, const avk::buffer& aOutTransferSource, const avk::buffer& aOutTransferTarget, const avk::buffer& aOutTransferTimeLeft, const avk::buffer& aInOutTransferring, const avk::buffer& aOutDeleteParticleList, const avk::buffer& aOutDeleteTransferList, const avk::buffer& aInTransferLength, const avk::buffer& aInOutDeleteParticleListLength, const avk::buffer& aInOutDeleteTransferListLength, float aDeltaTime);

	static void apply_acceleration(const avk::buffer& aInIndexList, const avk::buffer& aInOutVelocity, const avk::buffer& aInIndexListLength, const glm::vec3& aAcceleration);
	static void apply_velocity(const avk::buffer& aInIndexList, const avk::buffer& aInVelocity, const avk::buffer& aInOutPosition, const avk::buffer& aInIndexListLength, float aDeltaTime);
	static void infer_velocity(const avk::buffer& aInIndexList, const avk::buffer& aInOldPosition, const avk::buffer& aInPosition, const avk::buffer& aOutVelocity, const avk::buffer& aInIndexListLength, float aDeltaTime);

	static void sync_after_compute();
	static void sync_after_transfer();
private:
	static avk::descriptor_cache& descriptor_cache();
	static const avk::buffer& workgroup_count_buffer();
	static const avk::buffer& length_result_buffer();
	static void dispatch(uint32_t aX = 1u, uint32_t aY = 1u, uint32_t aZ = 1u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aOffset = 0u, uint32_t aScalingFactor = 1u, uint32_t aMinThreadCount = 0u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void dispatch_indirect();

	static avk::command_buffer mCmdBfr;
	static avk::queue* mQueue;
};
