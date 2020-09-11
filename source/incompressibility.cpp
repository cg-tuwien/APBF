#include "incompressibility.h"

pbd::incompressibility& pbd::incompressibility::set_data(fluid* aFluid, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors, gpu_list<8>* aNeighbors2)
{
	mFluid = aFluid;
	mNeighbors = aNeighbors;
	mNeighbors2 = aNeighbors2;
	return *this;
}

void pbd::incompressibility::apply()
{
	auto& kernelWidthList      = mFluid->get<fluid::id::kernel_width>();
	auto& boundarinessList     = mFluid->get<fluid::id::boundariness>();
	auto& particleList         = mFluid->get<fluid::id::particle>();
	auto& positionList         = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList      = particleList.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList           = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto  incompDataList       = pbd::gpu_list<32>().request_length(mFluid->requested_length());
	auto  scaledGradientList   = pbd::gpu_list<16>().request_length(mNeighbors2->requested_length());
	auto  lambdaList           = pbd::gpu_list< 4>().request_length(mFluid->requested_length());

//	shader_provider::incompressibility(particleList.index_buffer(), oldPositionList.buffer(), radiusList.buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), mNeighbors->buffer(), boundarinessList.write().buffer(), positionList.write().buffer(), mFluid->length());
	shader_provider::incompressibility_0(particleList.index_buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), incompDataList.write().buffer(), particleList.length());
	shader_provider::incompressibility_1(particleList.index_buffer(), positionList.buffer(), radiusList.buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), mNeighbors2->buffer(), incompDataList.write().buffer(), scaledGradientList.write().buffer(), mNeighbors2->length());
	shader_provider::incompressibility_2(particleList.index_buffer(), positionList.write().buffer(), radiusList.buffer(), inverseMassList.buffer(), incompDataList.buffer(), boundarinessList.write().buffer(), lambdaList.write().buffer(), particleList.length());
	shader_provider::incompressibility_3(particleList.index_buffer(), inverseMassList.buffer(), mNeighbors2->buffer(), scaledGradientList.buffer(), lambdaList.buffer(), positionList.write().buffer(), mNeighbors2->length());
}
