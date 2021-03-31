#include "neighborhood_brute_force.h"
#include "settings.h"

pbd::neighborhood_brute_force& pbd::neighborhood_brute_force::set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, neighbors* aNeighbors)
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
	if (settings::neighborListSorted) {
		mNeighbors->set_length(mParticles->length());
	} else {
		mNeighbors->set_length(0);
	}

	shader_provider::neighborhood_brute_force(mParticles->index_buffer(), positionList.buffer(), mRange->buffer(), mNeighbors->write().buffer(), mParticles->length(), mNeighbors->write().length(), mRangeScale);

	if (settings::neighborListSorted) {
		auto neighborCount = gpu_list<4>().request_length(mParticles->requested_length());
		auto prefixHelper  = gpu_list<4>().request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(neighborCount.requested_length()));
		auto linkedList    = *mNeighbors;

		shader_provider::copy_with_differing_stride(linkedList.buffer(), neighborCount.write().buffer(), mParticles->length(), 8u, 4u);
		algorithms::prefix_sum(neighborCount.write().buffer(), prefixHelper.write().buffer(), mParticles->length(), neighborCount.requested_length());
		shader_provider::linked_list_to_neighbor_list(linkedList.buffer(), neighborCount.buffer(), mNeighbors->write().buffer(), mParticles->length(), mNeighbors->write().length());
	}
}
