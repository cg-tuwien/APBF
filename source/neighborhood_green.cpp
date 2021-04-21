#include "neighborhood_green.h"
#include "measurements.h"
#include "settings.h"

pbd::neighborhood_green& pbd::neighborhood_green::set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, pbd::neighbors* aNeighbors)
{
	mParticles = aParticles;
	mRange = aRange;
	mNeighbors = aNeighbors;
	return *this;
}

pbd::neighborhood_green& pbd::neighborhood_green::set_range_scale(float aScale)
{
	mRangeScale = aScale;
	return *this;
}

pbd::neighborhood_green& pbd::neighborhood_green::set_position_range(const glm::vec3& aMinPos, const glm::vec3& aMaxPos, uint32_t aResolutionLog2)
{
	mMinPos = aMinPos;
	mMaxPos = aMaxPos;
	mResolutionLog2 = aResolutionLog2;
	return *this;
}

void pbd::neighborhood_green::apply()
{
	measurements::debug_label_start("neighborhood Green", glm::vec4(1, 0.5, 0, 1));

	auto  maxHash           = 1u << (mResolutionLog2 * DIMENSIONS);
	auto& positionList      = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto  unsortedHashList  = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortedHashList    = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  unsortedIndexList = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortedIndexList   = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortHelper        = pbd::gpu_list<sizeof(uint32_t)>().request_length(algorithms::sort_calculate_needed_helper_list_length(unsortedHashList.requested_length()));
	auto& cellStartList     = unsortedHashList;  // re-use lists
	auto& cellEndList       = unsortedIndexList;

	sortedIndexList.set_length(positionList.length());

	if (settings::neighborListSorted) {
		mNeighbors->set_length(mParticles->length());
	}
	else {
		mNeighbors->set_length(0);
	}

	// TODO only hash particles selected by index list!
	shader_provider::write_sequence(unsortedIndexList.write().buffer(), positionList.length(), 0u, 1u);
	shader_provider::calculate_position_hash(positionList.buffer(), unsortedHashList.write().buffer(), positionList.length(), mMinPos, mMaxPos, mResolutionLog2);
	algorithms::sort(unsortedHashList.write().buffer(), unsortedIndexList.write().buffer(), sortHelper.write().buffer(), positionList.length(), unsortedHashList.requested_length(), sortedHashList.write().buffer(), sortedIndexList.write().buffer(), maxHash);
	mParticles->hidden_list().apply_edit(sortedIndexList, nullptr); // sort particles according to hash (z-curve)

	mParticles->sort();

	cellStartList.set_length(maxHash);
	cellEndList  .set_length(maxHash);

	shader_provider::write_sequence(cellStartList.write().buffer(), cellStartList.write().length(), 0u, 0u);
	shader_provider::write_sequence(  cellEndList.write().buffer(),   cellEndList.write().length(), 0u, 0u);
	shader_provider::find_value_ranges(mParticles->index_buffer(), sortedHashList.buffer(), cellStartList.write().buffer(), cellEndList.write().buffer(), mParticles->length());
	shader_provider::neighborhood_green(mParticles->index_buffer(), positionList.buffer(), mRange->buffer(), cellStartList.buffer(), cellEndList.buffer(), mNeighbors->write().buffer(), mParticles->length(), mNeighbors->write().length(), mRangeScale, mMinPos, mMaxPos, mResolutionLog2);

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
