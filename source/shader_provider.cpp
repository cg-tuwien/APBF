#include "shader_provider.h"

avk::command_buffer shader_provider::mCmdBfr;
avk::queue* shader_provider::mQueue = nullptr;

void shader_provider::set_queue(avk::queue& aQueue)
{
	mQueue = &aQueue;
}

void shader_provider::start_recording()
{
	// Get a command pool to allocate command buffers from:
	auto& commandPool = gvk::context().get_command_pool_for_single_use_command_buffers(*mQueue);

	// Create a command buffer and render into the *current* swap chain image:
	mCmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	cmd_bfr()->begin_recording();
}

void shader_provider::end_recording()
{
	cmd_bfr()->end_recording();
	mQueue->submit(cmd_bfr());
	gvk::context().main_window()->handle_lifetime(std::move(mCmdBfr));
}

avk::command_buffer& shader_provider::cmd_bfr()
{
	return mCmdBfr;
}

void shader_provider::roundandround(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, const avk::top_level_acceleration_structure_t& aTlas, uint32_t aParticleCount)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/roundandround.comp",
		avk::descriptor_binding(0, 0, aAppData),
		avk::descriptor_binding(1, 0, aParticles),
		avk::descriptor_binding(1, 1, aAabbs->as_storage_buffer()),
		avk::descriptor_binding(2, 0, aTlas)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aAppData),
		avk::descriptor_binding(1, 0, aParticles),
		avk::descriptor_binding(1, 1, aAabbs->as_storage_buffer()),
		avk::descriptor_binding(2, 0, aTlas)
	}));
	dispatch(aParticleCount);
}

void shader_provider::mask_neighborhood(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, const avk::top_level_acceleration_structure_t& aTlas, uint32_t aParticleCount)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/mask_neighborhood.comp",
		avk::descriptor_binding(0, 0, aAppData),
		avk::descriptor_binding(1, 0, aParticles),
		avk::descriptor_binding(1, 1, aAabbs->as_storage_buffer()),
		avk::descriptor_binding(2, 0, aTlas)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aAppData),
		avk::descriptor_binding(1, 0, aParticles),
		avk::descriptor_binding(1, 1, aAabbs->as_storage_buffer()),
		avk::descriptor_binding(2, 0, aTlas)
	}));
	dispatch(aParticleCount);
}

const avk::buffer& shader_provider::append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const avk::buffer& aTargetListLength, const avk::buffer& aAppendingListLength, uint32_t aStride)
{
	struct push_constants { uint32_t mStride; } pushConstants{ aStride / 4 };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/append_list.comp",
		avk::descriptor_binding(0, 0, aTargetList),
		avk::descriptor_binding(0, 1, aAppendingList),
		avk::descriptor_binding(1, 0, aTargetListLength),
		avk::descriptor_binding(1, 1, aAppendingListLength),
		avk::descriptor_binding(1, 2, length_result_buffer()),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aAppendingListLength, 0u, aStride);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aTargetList),
		avk::descriptor_binding(0, 1, aAppendingList),
		avk::descriptor_binding(1, 0, aTargetListLength),
		avk::descriptor_binding(1, 1, aAppendingListLength),
		avk::descriptor_binding(1, 2, length_result_buffer())
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
	return length_result_buffer();
}

void shader_provider::copy_scattered_read(const avk::buffer& aSourceList, const avk::buffer& aTargetList, const avk::buffer& aEditList, const avk::buffer& aEditListLength, uint32_t aStride)
{
	struct push_constants { uint32_t mStride; } pushConstants{ aStride / 4 };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/copy_scattered_read.comp",
		avk::descriptor_binding(0, 0, aSourceList),
		avk::descriptor_binding(0, 1, aTargetList),
		avk::descriptor_binding(0, 2, aEditList),
		avk::descriptor_binding(1, 0, aEditListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aEditListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aSourceList),
		avk::descriptor_binding(0, 1, aTargetList),
		avk::descriptor_binding(0, 2, aEditList),
		avk::descriptor_binding(1, 0, aEditListLength),
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::scattered_write(const avk::buffer& aInIndexList, const avk::buffer& aOutBuffer, const avk::buffer& aInIndexListLength, uint32_t aValue)
{
	struct push_constants { uint32_t mValue; } pushConstants{ aValue };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/scattered_write.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::write_sequence(const avk::buffer& aOutBuffer, const avk::buffer& aInBufferLength, uint32_t aStartValue, uint32_t aSequenceValueStep)
{
	struct push_constants { uint32_t mStartValue, mSequenceValueStep; } pushConstants{ aStartValue, aSequenceValueStep };
	prepare_dispatch_indirect(aInBufferLength);
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/write_sequence.comp",
		avk::descriptor_binding(0, 0, aOutBuffer),
		avk::descriptor_binding(1, 0, aInBufferLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aOutBuffer),
		avk::descriptor_binding(1, 0, aInBufferLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

const avk::buffer& shader_provider::write_increasing_sequence(const avk::buffer& aTargetList, const avk::buffer& aNewTargetListLength, const avk::buffer& aSequenceMinValue, uint32_t aValueUpperBound, uint32_t aSequenceLength)
{
	struct push_constants { uint32_t mValueUpperBound, mSequenceLength; } pushConstants{ aValueUpperBound, aSequenceLength };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/write_increasing_sequence.comp",
		avk::descriptor_binding(0, 0, aTargetList),
		avk::descriptor_binding(1, 0, aSequenceMinValue),
		avk::descriptor_binding(1, 1, length_result_buffer()),
		avk::descriptor_binding(1, 2, aNewTargetListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aTargetList),
		avk::descriptor_binding(1, 0, aSequenceMinValue),
		avk::descriptor_binding(1, 1, length_result_buffer()),
		avk::descriptor_binding(1, 2, aNewTargetListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch(aSequenceLength);
	return length_result_buffer();
}

void shader_provider::find_value_ranges(const avk::buffer& aInBuffer, const avk::buffer& aOutRangeStart, const avk::buffer& aOutRangeEnd, const avk::buffer& aInBufferLength)
{
	prepare_dispatch_indirect(aInBufferLength);
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/find_value_ranges.comp",
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutRangeStart),
		avk::descriptor_binding(0, 2, aOutRangeEnd),
		avk::descriptor_binding(1, 0, aInBufferLength)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutRangeStart),
		avk::descriptor_binding(0, 2, aOutRangeEnd),
		avk::descriptor_binding(1, 0, aInBufferLength)
	}));
	dispatch_indirect();
}

void shader_provider::find_value_changes(const avk::buffer& aInBuffer, const avk::buffer& aOutChange, const avk::buffer& aInBufferLength, const avk::buffer& aOutChangeLength)
{
	prepare_dispatch_indirect(aInBufferLength);
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/find_value_changes.comp",
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutChange),
		avk::descriptor_binding(1, 0, aInBufferLength),
		avk::descriptor_binding(1, 1, aOutChangeLength)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutChange),
		avk::descriptor_binding(1, 0, aInBufferLength),
		avk::descriptor_binding(1, 1, aOutChangeLength)
	}));
	dispatch_indirect();
}

void shader_provider::indexed_subtract(const avk::buffer& aInIndexList, const avk::buffer& aInMinuend, const avk::buffer& aInSubtrahend, const avk::buffer& aOutDifference, const avk::buffer& aInIndexListLength)
{
	prepare_dispatch_indirect(aInIndexListLength);
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/indexed_subtract.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInMinuend),
		avk::descriptor_binding(0, 2, aInSubtrahend),
		avk::descriptor_binding(0, 3, aOutDifference),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInMinuend),
		avk::descriptor_binding(0, 2, aInSubtrahend),
		avk::descriptor_binding(0, 3, aOutDifference),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	dispatch_indirect();
}

void shader_provider::generate_new_index_list(const avk::buffer& aInRangeEnd, const avk::buffer& aOutBuffer, const avk::buffer& aInRangeEndLength, const avk::buffer& aOutBufferLength)
{
	prepare_dispatch_indirect(aInRangeEndLength);
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/generate_new_index_list.comp",
		avk::descriptor_binding(0, 0, aInRangeEnd),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(1, 0, aInRangeEndLength),
		avk::descriptor_binding(1, 1, aOutBufferLength)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInRangeEnd),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(1, 0, aInRangeEndLength),
		avk::descriptor_binding(1, 1, aOutBufferLength)
	}));
	dispatch_indirect();
}

void shader_provider::generate_new_edit_list(const avk::buffer& aInIndexList, const avk::buffer& aInEditList, const avk::buffer& aInRangeStart, const avk::buffer& aInTargetIndex, const avk::buffer& aOutBuffer, const avk::buffer& aInIndexListLength)
{
	prepare_dispatch_indirect(aInIndexListLength);
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/generate_new_edit_list.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInEditList),
		avk::descriptor_binding(0, 2, aInRangeStart),
		avk::descriptor_binding(0, 3, aInTargetIndex),
		avk::descriptor_binding(0, 4, aOutBuffer),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInEditList),
		avk::descriptor_binding(0, 2, aInRangeStart),
		avk::descriptor_binding(0, 3, aInTargetIndex),
		avk::descriptor_binding(0, 4, aOutBuffer),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	dispatch_indirect();
}

void shader_provider::prefix_sum_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aOutGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth)
{
	struct push_constants { uint32_t mLengthsAndOffsetsOffset, mRecursionDepth; } pushConstants{ aLengthsAndOffsetsOffset, aRecursionDepth };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/prefix_sum_apply_on_block_level.comp",
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aOutGroupSumBuffer),
		avk::descriptor_binding(0, 3, aLengthsAndOffsets),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aLengthsAndOffsets, aLengthsAndOffsetsOffset + aRecursionDepth, 1u, 512u);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aOutGroupSumBuffer),
		avk::descriptor_binding(0, 3, aLengthsAndOffsets)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::prefix_sum_spread_from_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aRecursionDepth)
{
	struct push_constants { uint32_t mLengthsAndOffsetsOffset, mRecursionDepth; } pushConstants{ aLengthsAndOffsetsOffset, aRecursionDepth };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/prefix_sum_spread_from_block_level.comp",
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aInGroupSumBuffer),
		avk::descriptor_binding(0, 3, aLengthsAndOffsets),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aLengthsAndOffsets, aLengthsAndOffsetsOffset + aRecursionDepth, 1u, 512u);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aInGroupSumBuffer),
		avk::descriptor_binding(0, 3, aLengthsAndOffsets)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::radix_sort_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aOutHistogramTable, const avk::buffer& aBufferLength, const avk::buffer& aLengthsAndOffsets, uint32_t aLengthsAndOffsetsOffset, uint32_t aSubkeyOffset, uint32_t aSubkeyLength)
{
	struct push_constants { uint32_t mLengthsAndOffsetsOffset, mSubkeyOffset, mSubkeyLength; } pushConstants{ aLengthsAndOffsetsOffset, aSubkeyOffset, aSubkeyLength };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/radix_sort_apply_on_block_level.comp",
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aInSecondBuffer),
		avk::descriptor_binding(0, 3, aOutSecondBuffer),
		avk::descriptor_binding(0, 4, aOutHistogramTable),
		avk::descriptor_binding(0, 5, aLengthsAndOffsets),
		avk::descriptor_binding(1, 0, aBufferLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aBufferLength, 0u, 1u, 512u);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aInSecondBuffer),
		avk::descriptor_binding(0, 3, aOutSecondBuffer),
		avk::descriptor_binding(0, 4, aOutHistogramTable),
		avk::descriptor_binding(0, 5, aLengthsAndOffsets),
		avk::descriptor_binding(1, 0, aBufferLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::radix_sort_scattered_write(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInSecondBuffer, const avk::buffer& aOutSecondBuffer, const avk::buffer& aInHistogramTable, const avk::buffer& aBufferLength, uint32_t aSubkeyOffset, uint32_t aSubkeyLength)
{
	struct push_constants { uint32_t mSubkeyOffset, mSubkeyLength; } pushConstants{ aSubkeyOffset, aSubkeyLength };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/radix_sort_scattered_write.comp",
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aInSecondBuffer),
		avk::descriptor_binding(0, 3, aOutSecondBuffer),
		avk::descriptor_binding(0, 4, aInHistogramTable),
		avk::descriptor_binding(1, 0, aBufferLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aBufferLength, 0u, 1u, 512u);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInBuffer),
		avk::descriptor_binding(0, 1, aOutBuffer),
		avk::descriptor_binding(0, 2, aInSecondBuffer),
		avk::descriptor_binding(0, 3, aOutSecondBuffer),
		avk::descriptor_binding(0, 4, aInHistogramTable),
		avk::descriptor_binding(1, 0, aBufferLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::initialize_box(const avk::buffer& aInIndexList, const avk::buffer& aInIndexListLength, const avk::buffer& aOutPosition, const avk::buffer& aOutVelocity, const avk::buffer& aOutInverseMass, const avk::buffer& aOutRadius, const glm::vec3& aMinPos, const glm::uvec3& aParticleCount, float aRadius, float aInverseMass, const glm::vec3& aVelocity)
{
	struct push_constants { glm::vec3 mMinPos; float mRadius; glm::vec3 mVelocity; float mInverseMass; glm::uvec3 mParticleCount; } pushConstants{ aMinPos, aRadius, aVelocity, aInverseMass, aParticleCount };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/initialize_box.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aOutPosition),
		avk::descriptor_binding(0, 2, aOutVelocity),
		avk::descriptor_binding(0, 3, aOutInverseMass),
		avk::descriptor_binding(0, 4, aOutRadius),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aOutPosition),
		avk::descriptor_binding(0, 2, aOutVelocity),
		avk::descriptor_binding(0, 3, aOutInverseMass),
		avk::descriptor_binding(0, 4, aOutRadius),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch(aParticleCount.x * aParticleCount.y * aParticleCount.z);
}

void shader_provider::add_box(const avk::buffer& aInIndexList, const avk::buffer& aOutBoxes, const glm::vec4& aMin, const glm::vec4& aMax)
{
	struct push_constants { glm::vec4 mMin, mMax; } pushConstants{ aMin, aMax };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/add_box.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aOutBoxes),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aOutBoxes)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch(1u, 1u, 1u, 1u);
}

void shader_provider::box_collision(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInBoxes, const avk::buffer& aInIndexListLength, const avk::buffer& aInBoxesLength)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/box_collision.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOutPosition),
		avk::descriptor_binding(0, 2, aInRadius),
		avk::descriptor_binding(0, 3, aInBoxes),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::descriptor_binding(1, 1, aInBoxesLength)
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOutPosition),
		avk::descriptor_binding(0, 2, aInRadius),
		avk::descriptor_binding(0, 3, aInBoxes),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::descriptor_binding(1, 1, aInBoxesLength)
	}));
	dispatch_indirect();
}

void shader_provider::neighborhood_brute_force(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRange, const avk::buffer& aOutNeighbors, const avk::buffer& aInIndexListLength, float aRangeScale)
{
	struct push_constants { float mRangeScale; } pushConstants{ aRangeScale };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/neighborhood_brute_force.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInPosition),
		avk::descriptor_binding(0, 2, aInRange),
		avk::descriptor_binding(0, 3, aOutNeighbors),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInPosition),
		avk::descriptor_binding(0, 2, aInRange),
		avk::descriptor_binding(0, 3, aOutNeighbors),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::inter_particle_collision(const avk::buffer& aInIndexList, const avk::buffer& aInOutPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInNeighbors, const avk::buffer& aInIndexListLength)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/inter_particle_collision.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOutPosition),
		avk::descriptor_binding(0, 2, aInRadius),
		avk::descriptor_binding(0, 3, aInInverseMass),
		avk::descriptor_binding(0, 4, aInNeighbors),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOutPosition),
		avk::descriptor_binding(0, 2, aInRadius),
		avk::descriptor_binding(0, 3, aInInverseMass),
		avk::descriptor_binding(0, 4, aInNeighbors),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	dispatch_indirect();
}

void shader_provider::incompressibility(const avk::buffer& aInIndexList, const avk::buffer& aInPosition, const avk::buffer& aInRadius, const avk::buffer& aInInverseMass, const avk::buffer& aInKernelWidth, const avk::buffer& aInNeighbors, const avk::buffer& aOutPosition, const avk::buffer& aInIndexListLength)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/incompressibility.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInPosition),
		avk::descriptor_binding(0, 2, aInRadius),
		avk::descriptor_binding(0, 3, aInInverseMass),
		avk::descriptor_binding(0, 4, aInKernelWidth),
		avk::descriptor_binding(0, 5, aInNeighbors),
		avk::descriptor_binding(0, 6, aOutPosition),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInPosition),
		avk::descriptor_binding(0, 2, aInRadius),
		avk::descriptor_binding(0, 3, aInInverseMass),
		avk::descriptor_binding(0, 4, aInKernelWidth),
		avk::descriptor_binding(0, 5, aInNeighbors),
		avk::descriptor_binding(0, 6, aOutPosition),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	dispatch_indirect();
}

void shader_provider::apply_acceleration(const avk::buffer& aInIndexList, const avk::buffer& aInOutVelocity, const avk::buffer& aInIndexListLength, const glm::vec3& aAcceleration)
{
	struct push_constants { glm::vec3 mAcceleration; } pushConstants{ aAcceleration };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/apply_acceleration.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOutVelocity),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOutVelocity),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::apply_velocity(const avk::buffer& aInIndexList, const avk::buffer& aInVelocity, const avk::buffer& aInOutPosition, const avk::buffer& aInIndexListLength, float aDeltaTime)
{
	struct push_constants { float mDeltaTime; } pushConstants{ aDeltaTime };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/apply_velocity.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInVelocity),
		avk::descriptor_binding(0, 2, aInOutPosition),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInVelocity),
		avk::descriptor_binding(0, 2, aInOutPosition),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::infer_velocity(const avk::buffer& aInIndexList, const avk::buffer& aInOldPosition, const avk::buffer& aInPosition, const avk::buffer& aOutVelocity, const avk::buffer& aInIndexListLength, float aDeltaTime)
{
	struct push_constants { float mDeltaTime; } pushConstants{ aDeltaTime };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/particle manipulation/infer_velocity.comp",
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOldPosition),
		avk::descriptor_binding(0, 2, aInPosition),
		avk::descriptor_binding(0, 3, aOutVelocity),
		avk::descriptor_binding(1, 0, aInIndexListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aInIndexListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aInIndexList),
		avk::descriptor_binding(0, 1, aInOldPosition),
		avk::descriptor_binding(0, 2, aInPosition),
		avk::descriptor_binding(0, 3, aOutVelocity),
		avk::descriptor_binding(1, 0, aInIndexListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::sync_after_compute()
{
	cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::compute_shader,                      /* -> */ avk::pipeline_stage::compute_shader,
		avk::memory_access::shader_buffers_and_images_any_access, /* -> */ avk::memory_access::shader_buffers_and_images_any_access
	);
	cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::compute_shader,                      /* -> */ avk::pipeline_stage::transfer,
		avk::memory_access::shader_buffers_and_images_any_access, /* -> */ avk::memory_access::transfer_any_access
	);
}

void shader_provider::sync_after_transfer()
{
	cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::transfer,           /* -> */ avk::pipeline_stage::compute_shader,
		avk::memory_access::transfer_any_access, /* -> */ avk::memory_access::shader_buffers_and_images_any_access
	);
	cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::transfer,           /* -> */ avk::pipeline_stage::transfer,
		avk::memory_access::transfer_any_access, /* -> */ avk::memory_access::transfer_any_access
	);
}

avk::descriptor_cache& shader_provider::descriptor_cache()
{
	static auto descriptorCache = gvk::context().create_descriptor_cache();
	return descriptorCache;
}

const avk::buffer& shader_provider::workgroup_count_buffer()
{
	static avk::buffer workgroupCount = gvk::context().create_buffer(
		avk::memory_usage::device, vk::BufferUsageFlagBits::eIndirectBuffer,
		avk::storage_buffer_meta::create_from_size(12)
	);
	return workgroupCount;
}

const avk::buffer& shader_provider::length_result_buffer()
{
	static avk::buffer lengthResult = gvk::context().create_buffer(
		avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
		avk::storage_buffer_meta::create_from_size(4)
	);
	return lengthResult;
}

void shader_provider::dispatch(uint32_t aX, uint32_t aY, uint32_t aZ, uint32_t aLocalSizeX, uint32_t aLocalSizeY, uint32_t aLocalSizeZ)
{
	cmd_bfr()->handle().dispatch((aX + aLocalSizeX - 1u) / aLocalSizeX, (aY + aLocalSizeY - 1u) / aLocalSizeY, (aZ + aLocalSizeZ - 1u) / aLocalSizeZ);
	sync_after_compute();
}

void shader_provider::prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aOffset, uint32_t aScalingFactor, uint32_t aLocalSizeX, uint32_t aLocalSizeY, uint32_t aLocalSizeZ)
{
	struct push_constants { uint32_t mOffset, mScalingFactor, mX, mY, mZ; } pushConstants{ aOffset, aScalingFactor, aLocalSizeX, aLocalSizeY, aLocalSizeZ };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/dispatch_indirect.comp",
		avk::descriptor_binding(0, 0, aXyz),
		avk::descriptor_binding(1, 0, workgroup_count_buffer()),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::descriptor_binding(0, 0, aXyz),
		avk::descriptor_binding(1, 0, workgroup_count_buffer())
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch(1u, 1u, 1u, 1u);
}

void shader_provider::dispatch_indirect()
{
	cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::compute_shader,                      /* -> */ avk::pipeline_stage::draw_indirect,
		avk::memory_access::shader_buffers_and_images_any_access, /* -> */ avk::memory_access::indirect_command_data_read_access
	);
	cmd_bfr()->handle().dispatchIndirect(workgroup_count_buffer()->buffer_handle(), 0);
	sync_after_compute();
}
