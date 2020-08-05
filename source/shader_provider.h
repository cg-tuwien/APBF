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
	static const avk::buffer& write_increasing_sequence(const avk::buffer& aTargetList, const avk::buffer& aNewTargetListLength, const avk::buffer& aSequenceMinValue, uint32_t aValueUpperBound, uint32_t aSequenceLength);
	static void find_value_ranges(const avk::buffer& aInBuffer, const avk::buffer& aOutRangeStart, const avk::buffer& aOutRangeEnd, const avk::buffer& aInBufferLength);
	static void find_value_changes(const avk::buffer& aInBuffer, const avk::buffer& aOutChange, const avk::buffer& aInBufferLength, const avk::buffer& aOutChangeLength);
	static void indexed_subtract(const avk::buffer& aInIndexList, const avk::buffer& aInMinuend, const avk::buffer& aInSubtrahend, const avk::buffer& aOutDifference, const avk::buffer& aInIndexListLength);
	static void generate_new_index_list(const avk::buffer& aInRangeEnd, const avk::buffer& aOutBuffer, const avk::buffer& aInRangeEndLength, const avk::buffer& aOutBufferLength);
	static void generate_new_edit_list(const avk::buffer& aInIndexList, const avk::buffer& aInEditList, const avk::buffer& aInRangeStart, const avk::buffer& aInTargetIndex, const avk::buffer& aOutBuffer, const avk::buffer& aInIndexListLength);
	static void prefix_sum_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aOutGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void prefix_sum_spread_from_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth);
	static void radix_sort_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aOutHistogramTable, const avk::buffer& aBufferLength, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);
	static void radix_sort_scattered_write(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aInHistogramTable, const avk::buffer& aBufferLength, uint32_t aSubkeyOffset, uint32_t aSubkeyLength);

	static void initialize_box(const avk::buffer& aInIndexList, const avk::buffer& aInIndexListLength, const avk::buffer& aOutPosition, const avk::buffer& aOutVelocity, const avk::buffer& aOutInverseMass, const avk::buffer& aOutRadius, const glm::vec3& aMinPos, const glm::uvec3& aParticleCount, float aRadius, float aInverseMass, const glm::vec3& aVelocity);

	static void add_box(const avk::buffer& aInIndexList, const avk::buffer& aOutBoxes, const glm::vec4& aMin, const glm::vec4& aMax);
	static void box_collision(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInBoxes, const avk::buffer& aInIndexListLength, const avk::buffer& aInBoxesLength);
	static void neighborhood_brute_force(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, float aRangeScale);
	static void inter_particle_collision(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInNeighbors, const avk::buffer& aInIndexListLength);
	static void incompressibility(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInKernelWidth, const avk::buffer& aInNeighbors, const avk::buffer& aOutPosition, const avk::buffer& aInIndexListLength);

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
	static void prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aOffset = 0u, uint32_t aScalingFactor = 1u, uint32_t aLocalSizeX = 256u, uint32_t aLocalSizeY = 1u, uint32_t aLocalSizeZ = 1u);
	static void dispatch_indirect();

	static avk::command_buffer mCmdBfr;
	static avk::queue* mQueue;
};
