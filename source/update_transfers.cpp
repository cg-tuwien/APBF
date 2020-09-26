#include "update_transfers.h"
#include "settings.h"
#include "../shaders/cpu_gpu_shared_config.h"

pbd::update_transfers& pbd::update_transfers::set_data(fluid* aFluid, neighbors* aNeighbors, transfers* aTransfers)
{
	mFluid = aFluid;
	mNeighbors = aNeighbors;
	mTransfers = aTransfers;
	return *this;
}

void pbd::update_transfers::apply()
{
	auto& targetRadiusList        = mFluid->get<fluid::id::target_radius>();
	auto& particleList            = mFluid->get<fluid::id::particle>();
	auto& boundarinessList        = mFluid->get<fluid::id::boundariness>();
	auto& boundaryDistanceList    = mFluid->get<fluid::id::boundary_distance>();
	auto& transferringList        = particleList.hidden_list().get<pbd::hidden_particles::id::transferring>();
	auto& positionList            = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& radiusList              = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& inverseMassList         = particleList.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& transferSourceList      = mTransfers->hidden_list().get<hidden_transfers::id::source>();
	auto& transferTargetList      = mTransfers->hidden_list().get<hidden_transfers::id::target>();
	auto& transferTimeLeftList    = mTransfers->hidden_list().get<hidden_transfers::id::time_left>();
	auto  oldBoundaryDistanceList = boundaryDistanceList;
	auto  splitList               = particles().share_hidden_data_from(particleList).request_length(particleList.requested_length());
	auto  timeLeftList            = gpu_list<4>().request_length(splitList.requested_length());
	auto  minNeighborDistList     = gpu_list<4>().request_length(mNeighbors->requested_length());
	auto  nearestNeighborList     = gpu_list<4>().request_length(mFluid->requested_length());

	shader_provider::write_sequence(minNeighborDistList.write().buffer(), mNeighbors->length(), std::numeric_limits<uint32_t>::max(), 0u);
	shader_provider::find_split_and_merge_1(mNeighbors->buffer(), particleList.index_buffer(), positionList.buffer(), oldBoundaryDistanceList.buffer(), boundaryDistanceList.write().buffer(), minNeighborDistList.write().buffer(), mNeighbors->length());
	shader_provider::find_split_and_merge_2(mNeighbors->buffer(), particleList.index_buffer(), positionList.buffer(), minNeighborDistList.buffer(), nearestNeighborList.write().buffer(), mNeighbors->length());
	shader_provider::find_split_and_merge_3(particleList.index_buffer(), positionList.buffer(), radiusList.buffer(), boundarinessList.buffer(), boundaryDistanceList.write().buffer(), targetRadiusList.write().buffer(), nearestNeighborList.buffer(), transferSourceList.write().index_buffer(), transferTargetList.write().index_buffer(), transferTimeLeftList.write().buffer(), transferringList.write().buffer(), splitList.write().index_buffer(), mFluid->length(), mTransfers->hidden_list().write().length(), splitList.write().length(), mTransfers->hidden_list().requested_length(), splitList.requested_length());

	// the new length of the transfers list was written into mTransfers->hidden_list().length(), which is the length buffer
	// of the first list in the uninterleaved_list bundle (the transfer source). It's important that at least the transfer
	// target list also has the correct length assigned, so that particle edits (e.g. deletion) affect the list correctly.
	mTransfers->hidden_list().set_length(mTransfers->hidden_list().length()); // TODO find cleaner solution

	// perform split

	if (settings::split) {
		splitList.set_length(shader_provider::remove_impossible_splits(splitList.index_buffer(), transferringList.write().buffer(), mTransfers->hidden_list().length(), particleList.hidden_list().length(), splitList.length(), mTransfers->requested_length(), particleList.hidden_list().requested_length()));
		timeLeftList.set_length(splitList.length());
		shader_provider::write_sequence_float(timeLeftList.write().buffer(), timeLeftList.write().length(), -SPLIT_DURATION, 0);
		auto duplicates = splitList.duplicate_these();
		shader_provider::initialize_split_particles(duplicates.index_buffer(), positionList.write().buffer(), inverseMassList.write().buffer(), radiusList.write().buffer(), duplicates.length());

		transferSourceList += splitList;
		transferTargetList += duplicates;
		transferTimeLeftList += timeLeftList;
	}
}
