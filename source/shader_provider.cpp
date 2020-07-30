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

void shader_provider::roundandround(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, uint32_t aParticleCount)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/roundandround.comp",
		avk::binding(0, 0, aAppData),
		avk::binding(1, 0, aParticles),
		avk::binding(1, 1, aAabbs->as_storage_buffer())
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aAppData),
		avk::binding(1, 0, aParticles),
		avk::binding(1, 1, aAabbs->as_storage_buffer())
	}));
	dispatch(aParticleCount);
}

void shader_provider::append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const changing_length& aTargetListLength, const avk::buffer& aAppendingListLength, uint32_t aStride)
{
	struct push_constants { uint32_t mStride; } pushConstants{ aStride / 4 };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/append_list.comp",
		avk::binding(0, 0, aTargetList),
		avk::binding(0, 1, aAppendingList),
		avk::binding(1, 0, aTargetListLength.mOldLength),
		avk::binding(1, 1, aAppendingListLength),
		avk::binding(1, 2, aTargetListLength.mNewLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aAppendingListLength, 0u, aStride);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aTargetList),
		avk::binding(0, 1, aAppendingList),
		avk::binding(1, 0, aTargetListLength.mOldLength),
		avk::binding(1, 1, aAppendingListLength),
		avk::binding(1, 2, aTargetListLength.mNewLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::copy_scattered_read(const avk::buffer& aSourceList, const avk::buffer& aTargetList, const avk::buffer& aEditList, const avk::buffer& aEditListLength, const avk::buffer& aNewTargetListLength, uint32_t aStride)
{
	struct push_constants { uint32_t mStride; } pushConstants{ aStride / 4 };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/copy_scattered_read.comp",
		avk::binding(0, 0, aSourceList),
		avk::binding(0, 1, aTargetList),
		avk::binding(0, 2, aEditList),
		avk::binding(1, 0, aEditListLength),
		avk::binding(1, 1, aNewTargetListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aEditListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aSourceList),
		avk::binding(0, 1, aTargetList),
		avk::binding(0, 2, aEditList),
		avk::binding(1, 0, aEditListLength),
		avk::binding(1, 1, aNewTargetListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::write_increasing_sequence(const avk::buffer& aTargetList, const avk::buffer& aNewTargetListLength, const changing_length& aSequenceMinValue, uint32_t aValueUpperBound, uint32_t aSequenceLength)
{
	struct push_constants { uint32_t mValueUpperBound; uint32_t mSequenceLength; } pushConstants{ aValueUpperBound, aSequenceLength };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/write_increasing_sequence.comp",
		avk::binding(0, 0, aTargetList),
		avk::binding(1, 0, aSequenceMinValue.mOldLength),
		avk::binding(1, 1, aSequenceMinValue.mNewLength),
		avk::binding(1, 2, aNewTargetListLength),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aTargetList),
		avk::binding(1, 0, aSequenceMinValue.mOldLength),
		avk::binding(1, 1, aSequenceMinValue.mNewLength),
		avk::binding(1, 2, aNewTargetListLength)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch(aSequenceLength);
}

void shader_provider::prefix_sum_apply_on_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aOutGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aRecursionDepth)
{
	struct push_constants { uint32_t mRecursionDepth; } pushConstants{ aRecursionDepth };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/prefix_sum_apply_on_block_level.comp",
		avk::binding(0, 0, aInBuffer),
		avk::binding(0, 1, aOutBuffer),
		avk::binding(0, 2, aOutGroupSumBuffer),
		avk::binding(0, 3, aLengthsAndOffsets),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aLengthsAndOffsets, aRecursionDepth, 1u, 512u);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aInBuffer),
		avk::binding(0, 1, aOutBuffer),
		avk::binding(0, 2, aOutGroupSumBuffer),
		avk::binding(0, 3, aLengthsAndOffsets)
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch_indirect();
}

void shader_provider::prefix_sum_spread_from_block_level(const avk::buffer& aInBuffer, const avk::buffer& aOutBuffer, const avk::buffer& aInGroupSumBuffer, const avk::buffer& aLengthsAndOffsets, uint32_t aRecursionDepth)
{
	struct push_constants { uint32_t mRecursionDepth; } pushConstants{ aRecursionDepth };
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/prefix_sum_spread_from_block_level.comp",
		avk::binding(0, 0, aInBuffer),
		avk::binding(0, 1, aOutBuffer),
		avk::binding(0, 2, aInGroupSumBuffer),
		avk::binding(0, 3, aLengthsAndOffsets),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	prepare_dispatch_indirect(aLengthsAndOffsets, aRecursionDepth, 1u, 512u);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aInBuffer),
		avk::binding(0, 1, aOutBuffer),
		avk::binding(0, 2, aInGroupSumBuffer),
		avk::binding(0, 3, aLengthsAndOffsets)
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
		avk::binding(0, 0, aXyz),
		avk::binding(1, 0, workgroup_count_buffer()),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, sizeof(pushConstants) }
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aXyz),
		avk::binding(1, 0, workgroup_count_buffer())
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
