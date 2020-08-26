#include "incompressibility.h"
#include "settings.h"

pbd::incompressibility& pbd::incompressibility::set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors)
{
	mFluid = aFluid;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::incompressibility::apply()
{
	auto& targetRadiusList     = mFluid->get<fluid::id::target_radius>();
	auto& kernelWidthList      = mFluid->get<fluid::id::kernel_width>();
	auto& boundarinessList     = mFluid->get<fluid::id::boundariness>();
	auto& particleList         = mFluid->get<fluid::id::particle>();
	auto& positionList         = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList      = particleList.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList           = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto oldPositionList       = positionList;
	auto uintKernelWidthList   = pbd::gpu_list<4>().request_length(kernelWidthList.requested_length());

	shader_provider::incompressibility(particleList.index_buffer(), oldPositionList.buffer(), radiusList.buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), mNeighbors->buffer(), boundarinessList.write().buffer(), positionList.write().buffer(), mFluid->length(), pbd::settings::updateBoundariness);
}
