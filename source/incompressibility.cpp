#include "incompressibility.h"

pbd::incompressibility& pbd::incompressibility::set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors, transfers* aTransfers)
{
	mFluid = aFluid;
	mNeighbors = aNeighbors;
	mTransfers = aTransfers;
	return *this;
}

void pbd::incompressibility::apply()
{
	auto& targetRadiusList     = mFluid->get<fluid::id::target_radius>();
	auto& boundaryDistList     = mFluid->get<fluid::id::boundary_distance>();
	auto& kernelWidthList      = mFluid->get<fluid::id::kernel_width>();
	auto& transferringList     = mFluid->get<fluid::id::transferring>();
	auto& particleList         = mFluid->get<fluid::id::particle>();
	auto& positionList         = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList      = particleList.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList           = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& transferSourceList   = mTransfers->hidden_list().get<hidden_transfers::id::source>();
	auto& transferTargetList   = mTransfers->hidden_list().get<hidden_transfers::id::target>();
	auto& transferTimeLeftList = mTransfers->hidden_list().get<hidden_transfers::id::time_left>();
	auto oldPositionList       = positionList;
	auto oldBoundaryDistList   = boundaryDistList;
	auto uintKernelWidthList   = pbd::gpu_list<4>().request_length(kernelWidthList.requested_length());

	shader_provider::write_sequence(uintKernelWidthList.write().buffer(), kernelWidthList.length(), 0, 0);
	shader_provider::kernel_width(particleList.index_buffer(), positionList.buffer(), radiusList.buffer(), targetRadiusList.buffer(), kernelWidthList.buffer(), uintKernelWidthList.write().buffer(), mNeighbors->write().buffer(), mFluid->length());
	shader_provider::incompressibility(particleList.index_buffer(), oldPositionList.buffer(), radiusList.buffer(), inverseMassList.buffer(), uintKernelWidthList.buffer(), kernelWidthList.write().buffer(), oldBoundaryDistList.buffer(), boundaryDistList.write().buffer(), targetRadiusList.write().buffer(), mNeighbors->buffer(), positionList.write().buffer(), transferSourceList.write().index_buffer(), transferTargetList.write().index_buffer(), transferTimeLeftList.write().buffer(), transferringList.write().buffer(), mFluid->length(), mTransfers->hidden_list().write().length());
}
