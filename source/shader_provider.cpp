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
//		avk::binding<avk::buffer>(0, 0, 1u),
//		avk::binding<avk::buffer>(1, 0, 1u),
//		avk::binding<avk::buffer>(1, 1, 1u)
		avk::binding(0, 0, aAppData),
		avk::binding(1, 0, aParticles),
		avk::binding(1, 1, aAabbs)
	);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aAppData),
		avk::binding(1, 0, aParticles),
		avk::binding(1, 1, aAabbs)
	}));
	dispatch(aParticleCount, 1u, 1u, 256u);
}

void shader_provider::append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const avk::buffer& aTargetListLength, const avk::buffer& aAppendingListLength)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/append_list.comp",
		avk::binding(0, 0, aTargetList),
		avk::binding(0, 1, aAppendingList),
		avk::binding(1, 0, aTargetListLength),
		avk::binding(1, 1, aAppendingListLength)
	);
	prepare_dispatch_indirect(aAppendingListLength);
	cmd_bfr()->bind_pipeline(pipeline);
	cmd_bfr()->bind_descriptors(pipeline->layout(), descriptor_cache().get_or_create_descriptor_sets({
		avk::binding(0, 0, aTargetList),
		avk::binding(0, 1, aAppendingList),
		avk::binding(1, 0, aTargetListLength),
		avk::binding(1, 1, aAppendingListLength)
	}));
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

void shader_provider::prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aLocalSizeX, uint32_t aLocalSizeY, uint32_t aLocalSizeZ)
{
	static auto pipeline = gvk::context().create_compute_pipeline_for(
		"shaders/dispatch_indirect.comp",
		avk::binding(0, 0, aXyz),
		avk::binding(1, 0, workgroup_count_buffer()),
		avk::push_constant_binding_data{ avk::shader_type::compute, 0, 12 }
	);
	struct push_constants { uint32_t mX, mY, mZ; } pushConstants{ aLocalSizeX, aLocalSizeY, aLocalSizeZ };
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
