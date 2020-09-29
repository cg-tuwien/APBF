#include "spread_kernel_width.h"
#include "settings.h"
#include "../shaders/cpu_gpu_shared_config.h"

pbd::spread_kernel_width& pbd::spread_kernel_width::set_data(fluid* aFluid, neighbors* aNeighbors)
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
	auto  uintKernelWidthList  = pbd::gpu_list<4>().request_length(kernelWidthList.requested_length());
	auto  oldNeighborsList     = *mNeighbors;
	mNeighbors->set_length(0);

	shader_provider::kernel_width_init(particleList.index_buffer(), radiusList.buffer(), targetRadiusList.buffer(), uintKernelWidthList.write().buffer(), particleList.length());
	shader_provider::kernel_width(particleList.index_buffer(), positionList.buffer(), radiusList.buffer(), targetRadiusList.buffer(), kernelWidthList.buffer(), uintKernelWidthList.write().buffer(), oldNeighborsList.buffer(), mNeighbors->write().buffer(), oldNeighborsList.length(), mNeighbors->write().length());
	shader_provider::uint_to_float_but_gradual(uintKernelWidthList.buffer(), kernelWidthList.write().buffer(), mFluid->length(), 1.0f / KERNEL_WIDTH_RESOLUTION, settings::kernelWidthAdaptionSpeed); // TODO should adaption speed be scaled by delta time?
}
