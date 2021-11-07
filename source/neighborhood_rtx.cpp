#include "neighborhood_rtx.h"
#include "measurements.h"
#include "settings.h"
#include "../shaders/cpu_gpu_shared_config.h"

pbd::neighborhood_rtx::neighborhood_rtx()
{
	mStepsUntilNextRebuild = 0u;
	mBlas = gvk::context().create_bottom_level_acceleration_structure({ avk::acceleration_structure_size_requirements::from_aabbs(1u) }, false, [](avk::bottom_level_acceleration_structure_t& blas) {
		//blas.config().flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild; // TODO: Test performance of FastTrace
	});
	mBlas->build({ VkAabbPositionsKHR{ /* min: */ -1.f, -1.f, -1.f,  /* max: */ 1.f,  1.f,  1.f } });
}

pbd::neighborhood_rtx& pbd::neighborhood_rtx::set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, neighbors* aNeighbors)
{
	mParticles = aParticles;
	mRange = aRange;
	mNeighbors = aNeighbors;
	if (!mTlas.has_value() || mMaxInstanceCount < mParticles->requested_length()) {
		mMaxInstanceCount = mParticles->requested_length();
		mTlas = gvk::context().create_top_level_acceleration_structure(static_cast<uint32_t>(mMaxInstanceCount), true, [](avk::top_level_acceleration_structure_t& tlas) {
			//tlas.config().flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild; // TODO: Test performance of FastTrace
		});
	}
	return *this;
}

pbd::neighborhood_rtx& pbd::neighborhood_rtx::set_range_scale(float aScale)
{
	mRangeScale = aScale;
	return *this;
}

void pbd::neighborhood_rtx::apply()
{
	measurements::debug_label_start("neighborhood RTX", glm::vec4(1, 0.5, 0, 1));

	auto& positionList     = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto  blasReference   = mBlas->device_address();

	if (settings::neighborListSorted) {
		mNeighbors->set_length(mParticles->length());
	}
	else {
		mNeighbors->set_length(0);
	}

	reserve_geometry_instances_buffer(mParticles->requested_length());
	shader_provider::generate_acceleration_structure_instances(mParticles->index_buffer(), positionList.buffer(), mRange->buffer(), mGeometryInstances, mParticles->length(), blasReference, mRangeScale, static_cast<uint32_t>(mMaxInstanceCount));
	build_acceleration_structure();
	shader_provider::neighborhood_rtx(mParticles->index_buffer(), positionList.buffer(), mRange->buffer(), mNeighbors->write().buffer(), mParticles->length(), mNeighbors->write().length(), mTlas, mRangeScale);

	if (settings::neighborListSorted) {
		auto neighborCount = gpu_list<4>().request_length(mParticles->requested_length());
		auto prefixHelper  = gpu_list<4>().request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(neighborCount.requested_length()));
		auto linkedList    = *mNeighbors;

		shader_provider::copy_with_differing_stride(linkedList.buffer(), neighborCount.write().buffer(), mParticles->length(), 8u, 4u);
		algorithms::prefix_sum(neighborCount.write().buffer(), prefixHelper.write().buffer(), mParticles->length(), neighborCount.requested_length());
		shader_provider::linked_list_to_neighbor_list(linkedList.buffer(), neighborCount.buffer(), mNeighbors->write().buffer(), mParticles->length(), mNeighbors->write().length());
	}

	measurements::debug_label_end();
}

void pbd::neighborhood_rtx::reserve_geometry_instances_buffer(size_t aSize)
{
	auto size = aSize * sizeof(vk::AccelerationStructureInstanceKHR);
	if (mGeometryInstances.has_value() && mGeometryInstances->meta_at_index<avk::buffer_meta>().total_size() >= size) return;

	mGeometryInstances = gvk::context().create_buffer(
		avk::memory_usage::device, {},
		avk::storage_buffer_meta::create_from_size(size),
		avk::geometry_instance_buffer_meta::create_from_num_elements(aSize, sizeof(vk::AccelerationStructureInstanceKHR))
	);
}

void pbd::neighborhood_rtx::build_acceleration_structure()
{
	shader_provider::cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::compute_shader,                        /* -> */ avk::pipeline_stage::acceleration_structure_build,
		avk::memory_access::shader_buffers_and_images_write_access, /* -> */ avk::memory_access::acceleration_structure_any_access
	);

	if (mStepsUntilNextRebuild-- == 0u) {
		mTlas->build(mGeometryInstances, {}, avk::sync::with_barriers_into_existing_command_buffer(*shader_provider::cmd_bfr(), {}, {}));
		mStepsUntilNextRebuild = FIXED_TIME_STEP == 0 ? 0u : 60u;
	} else {
		mTlas->update(mGeometryInstances, {}, avk::sync::with_barriers_into_existing_command_buffer(*shader_provider::cmd_bfr(), {}, {}));
	}

	shader_provider::cmd_bfr()->establish_global_memory_barrier(
		avk::pipeline_stage::acceleration_structure_build,       /* -> */ avk::pipeline_stage::compute_shader,
		avk::memory_access::acceleration_structure_write_access, /* -> */ avk::memory_access::shader_buffers_and_images_any_access
	);
}
