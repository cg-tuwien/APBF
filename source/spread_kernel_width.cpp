#include "spread_kernel_width.h"
#include "settings.h"

pbd::spread_kernel_width& pbd::spread_kernel_width::set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors)
{
	mFluid = aFluid;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::spread_kernel_width::apply()
{
	auto& targetRadiusList     = mFluid->get<fluid::id::target_radius>();
	auto& kernelWidthList      = mFluid->get<fluid::id::kernel_width>();
	auto& particleList         = mFluid->get<fluid::id::particle>();
	auto& positionList         = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& radiusList           = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto uintKernelWidthList   = pbd::gpu_list<4>().request_length(kernelWidthList.requested_length());

	shader_provider::write_sequence(uintKernelWidthList.write().buffer(), kernelWidthList.length(), 0, 0);
	shader_provider::kernel_width(particleList.index_buffer(), positionList.buffer(), radiusList.buffer(), targetRadiusList.buffer(), kernelWidthList.buffer(), uintKernelWidthList.write().buffer(), mNeighbors->write().buffer(), mFluid->length(), pbd::settings::baseKernelWidthOnTargetRadius);
	shader_provider::uint_to_float(uintKernelWidthList.buffer(), kernelWidthList.write().buffer(), mFluid->length(), 1.0f / KERNEL_WIDTH_RESOLUTION);
}
