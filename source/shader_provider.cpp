#include "shader_provider.h"

ak::command_buffer shader_provider::mCmdBfr;
ak::queue* shader_provider::mQueue = nullptr;

void shader_provider::set_queue(ak::queue& aQueue)
{
	mQueue = &aQueue;
}

void shader_provider::start_recording()
{
	// Get a command pool to allocate command buffers from:
	auto& commandPool = xk::context().get_command_pool_for_single_use_command_buffers(*mQueue);

	// Create a command buffer and render into the *current* swap chain image:
	mCmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	cmd_bfr()->begin_recording();
}

void shader_provider::end_recording()
{
	cmd_bfr()->end_recording();
	mQueue->submit(cmd_bfr());
	xk::context().main_window()->handle_lifetime(std::move(mCmdBfr));
}

ak::command_buffer& shader_provider::cmd_bfr()
{
	return mCmdBfr;
}

void shader_provider::roundandround(const ak::buffer& aAppData, const ak::buffer& aParticles, const ak::buffer& aAabbs, uint32_t aParticleCount)
{
	static auto pipeline = xk::context().create_compute_pipeline_for(
		"shaders/roundandround.comp",
//		ak::binding<ak::buffer>(0, 0, 1u),
//		ak::binding<ak::buffer>(1, 0, 1u),
//		ak::binding<ak::buffer>(1, 1, 1u)
		ak::binding(0, 0, aAppData),
		ak::binding(1, 0, aParticles),
		ak::binding(1, 1, aAabbs)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		ak::binding(0, 0, aAppData),
		ak::binding(1, 0, aParticles),
		ak::binding(1, 1, aAabbs)
	}));
	dispatch(aParticleCount, 1u, 1u, 256u);
}

void shader_provider::append_list(const ak::buffer& aTargetList, const ak::buffer& aAppendingList, const ak::buffer& aTargetListLength, const ak::buffer& aAppendingListLength)
{
	static auto pipeline = xk::context().create_compute_pipeline_for(
		"shaders/append_list.comp",
		ak::binding(0, 0, aTargetList),
		ak::binding(0, 1, aAppendingList),
		ak::binding(1, 0, aTargetListLength),
		ak::binding(1, 1, aAppendingListLength)
	);
	prepare_dispatch_indirect(aAppendingListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		ak::binding(0, 0, aTargetList),
		ak::binding(0, 1, aAppendingList),
		ak::binding(1, 0, aTargetListLength),
		ak::binding(1, 1, aAppendingListLength)
	}));
	dispatch_indirect();
}

void shader_provider::sync_after_compute()
{
	cmd_bfr()->establish_global_memory_barrier(
		ak::pipeline_stage::compute_shader,                      /* -> */ ak::pipeline_stage::compute_shader,
		ak::memory_access::shader_buffers_and_images_any_access, /* -> */ ak::memory_access::shader_buffers_and_images_any_access
	);
	cmd_bfr()->establish_global_memory_barrier(
		ak::pipeline_stage::compute_shader,                      /* -> */ ak::pipeline_stage::transfer,
		ak::memory_access::shader_buffers_and_images_any_access, /* -> */ ak::memory_access::transfer_any_access
	);
}

void shader_provider::sync_after_transfer()
{
	cmd_bfr()->establish_global_memory_barrier(
		ak::pipeline_stage::transfer,           /* -> */ ak::pipeline_stage::compute_shader,
		ak::memory_access::transfer_any_access, /* -> */ ak::memory_access::shader_buffers_and_images_any_access
	);
	cmd_bfr()->establish_global_memory_barrier(
		ak::pipeline_stage::transfer,           /* -> */ ak::pipeline_stage::transfer,
		ak::memory_access::transfer_any_access, /* -> */ ak::memory_access::transfer_any_access
	);
}

ak::descriptor_cache& shader_provider::descriptor_cache()
{
	static auto descriptorCache = xk::context().create_descriptor_cache();
	return descriptorCache;
}

const ak::buffer& shader_provider::workgroup_count_buffer()
{
	static ak::buffer workgroupCount = xk::context().create_buffer(
		ak::memory_usage::device, vk::BufferUsageFlagBits::eIndirectBuffer,
		ak::storage_buffer_meta::create_from_size(12)
	);
	return workgroupCount;
}

void shader_provider::dispatch(uint32_t aX, uint32_t aY, uint32_t aZ, uint32_t aLocalSizeX, uint32_t aLocalSizeY, uint32_t aLocalSizeZ)
{
	cmd_bfr()->handle().dispatch((aX + aLocalSizeX - 1u) / aLocalSizeX, (aY + aLocalSizeY - 1u) / aLocalSizeY, (aZ + aLocalSizeZ - 1u) / aLocalSizeZ);
	sync_after_compute();
}

void shader_provider::prepare_dispatch_indirect(const ak::buffer& aXyz, uint32_t aLocalSizeX, uint32_t aLocalSizeY, uint32_t aLocalSizeZ)
{
	static auto pipeline = xk::context().create_compute_pipeline_for(
		"shaders/dispatch_indirect.comp",
		ak::binding(0, 0, aXyz),
		ak::binding(1, 0, workgroup_count_buffer()),
		ak::push_constant_binding_data{ ak::shader_type::compute, 0, 12 }
	);
	struct push_constants { uint32_t mX, mY, mZ; } pushConstants{ aLocalSizeX, aLocalSizeY, aLocalSizeZ };
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		ak::binding(0, 0, aXyz),
		ak::binding(1, 0, workgroup_count_buffer())
	}));
	cmd_bfr()->push_constants(pipeline->layout(), pushConstants);
	dispatch(1u, 1u, 1u, 1u);
}

void shader_provider::dispatch_indirect()
{
	cmd_bfr()->establish_global_memory_barrier(
		ak::pipeline_stage::compute_shader,                      /* -> */ ak::pipeline_stage::draw_indirect,
		ak::memory_access::shader_buffers_and_images_any_access, /* -> */ ak::memory_access::indirect_command_data_read_access
	);
	cmd_bfr()->handle().dispatchIndirect(workgroup_count_buffer()->buffer_handle(), 0);
	sync_after_compute();
}
