#include "neighborhood_green.h"

pbd::neighborhood_green& pbd::neighborhood_green::set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors)
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
	auto  maxHash           = 1u << (mResolutionLog2 * DIMENSIONS);
	auto& positionList      = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto  unsortedHashList  = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortedHashList    = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  unsortedIndexList = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortedIndexList   = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortHelper        = pbd::gpu_list<sizeof(uint32_t)>().request_length(algorithms::sort_calculate_needed_helper_list_length(unsortedHashList.requested_length()));
	auto& cellStartList     = unsortedHashList;  // re-use lists
	auto  cellEndList       = unsortedIndexList;

	sortedIndexList.set_length(positionList.length());

	// TODO only hash particles selected by index list!
	shader_provider::write_sequence(unsortedIndexList.write().buffer(), positionList.length(), 0u, 1u);
	shader_provider::calculate_position_hash(positionList.buffer(), unsortedHashList.write().buffer(), positionList.length(), mMinPos, mMaxPos, mResolutionLog2);
	algorithms::sort(unsortedHashList.write().buffer(), unsortedIndexList.write().buffer(), sortHelper.write().buffer(), positionList.length(), sortedHashList.write().buffer(), sortedIndexList.write().buffer(), maxHash);
	mParticles->hidden_list().apply_edit(sortedIndexList, nullptr); // sort particles according to hash (z-curve)

	cellStartList.set_length(maxHash);
	cellEndList.set_length(maxHash);

	shader_provider::write_sequence(cellStartList.write().buffer(), cellStartList.write().length(), 0u, 0u);
	shader_provider::write_sequence(cellEndList.write().buffer(), cellEndList.write().length(), 0u, 0u);
	shader_provider::find_value_ranges(sortedHashList.buffer(), cellStartList.write().buffer(), cellEndList.write().buffer(), positionList.length());
	shader_provider::neighborhood_green(mParticles->index_buffer(), positionList.buffer(), mRange->buffer(), cellStartList.buffer(), cellEndList.buffer(), mNeighbors->write().buffer(), mParticles->length(), mRangeScale, mMinPos, mMaxPos, mResolutionLog2);
}
