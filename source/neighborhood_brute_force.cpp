#include "neighborhood_brute_force.h"

pbd::neighborhood_brute_force& pbd::neighborhood_brute_force::set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, gpu_list<sizeof(uint32_t) * 64>* aNeighbors)
{
	mParticles = aParticles;
	mRange = aRange;
	mNeighbors = aNeighbors;
	return *this;
}

pbd::neighborhood_brute_force& pbd::neighborhood_brute_force::set_range_scale(float aScale)
{
	mRangeScale = aScale;
	return *this;
}

void pbd::neighborhood_brute_force::apply()
{
	auto& positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();

	shader_provider::neighborhood_brute_force(mParticles->index_buffer(), positionList.buffer(), mRange->buffer(), mNeighbors->write().buffer(), mParticles->length(), mRangeScale);
}
