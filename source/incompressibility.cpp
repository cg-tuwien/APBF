#include "incompressibility.h"

pbd::incompressibility& pbd::incompressibility::set_data(particles* aParticles, gpu_list<sizeof(uint32_t) * 64>* aNeighbors)
{
	mParticles = aParticles;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::incompressibility::apply()
{
	auto& positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList = mParticles->hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& kernelWidthList = radiusList; // TODO

	shader_provider::incompressibility(mParticles->index_buffer(), positionList.write().buffer(), radiusList.buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), mNeighbors->buffer(), positionList.write().buffer(), mParticles->length());
}
