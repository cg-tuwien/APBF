#include "neighborhood_binary_search.h"

pbd::neighborhood_binary_search& pbd::neighborhood_binary_search::set_data(particles* aParticles, const gpu_list<sizeof(float)>* aRange, pbd::neighbors* aNeighbors)
{
	mParticles = aParticles;
	mRange = aRange;
	mNeighbors = aNeighbors;
	return *this;
}

pbd::neighborhood_binary_search& pbd::neighborhood_binary_search::set_range_scale(float aScale)
{
	mRangeScale = aScale;
	return *this;
}

void pbd::neighborhood_binary_search::apply()
{
	auto& positionList      = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto  unsortedCodeList  = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  unsortedIndexList = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortedIndexList   = pbd::gpu_list<sizeof(uint32_t)>().request_length(positionList.requested_length());
	auto  sortHelper        = pbd::gpu_list<sizeof(uint32_t)>().request_length(algorithms::sort_calculate_needed_helper_list_length(unsortedCodeList.requested_length()));
	auto  sortedCodeLists   = std::vector<pbd::gpu_list<sizeof(uint32_t)>>();
	
	sortedCodeLists.resize(DIMENSIONS);
	for (auto& l : sortedCodeLists) l.request_length(positionList.requested_length());

	sortedIndexList.set_length(positionList.length());
	mNeighbors    ->set_length(0);

	shader_provider::write_sequence(unsortedIndexList.write().buffer(), positionList.length(), 0u, 1u);

	for (auto i = 0u; i < DIMENSIONS; i++) {
		shader_provider::calculate_position_code(unsortedIndexList.buffer(), positionList.buffer(), unsortedCodeList.write().buffer(), positionList.length(), i);
		algorithms::sort(unsortedCodeList.write().buffer(), unsortedIndexList.write().buffer(), sortHelper.write().buffer(), positionList.length(), unsortedCodeList.requested_length(), sortedCodeLists[i].write().buffer(), sortedIndexList.write().buffer());
		unsortedIndexList = sortedIndexList;
	}

	mParticles->hidden_list().apply_edit(sortedIndexList, nullptr); // sort particles according to position code (z-curve)
	mParticles->sort();

	shader_provider::write_sequence(unsortedIndexList.write().buffer(), positionList.length(), 0u, 1u);

	for (auto i = 0u; i < DIMENSIONS; i++) {
		shader_provider::calculate_position_code(mParticles->index_buffer(), positionList.buffer(), sortedCodeLists[i].write().buffer(), positionList.length(), i);
	}

	shader_provider::neighborhood_binary_search(mParticles->index_buffer(), positionList.buffer(), sortedCodeLists[0].buffer(), sortedCodeLists[1].buffer(), sortedCodeLists[DIMENSIONS - 1].buffer(), mRange->buffer(), mNeighbors->write().buffer(), mParticles->length(), mNeighbors->write().length(), mRangeScale);
}
