#include "particle_transfer.h"

pbd::particle_transfer& pbd::particle_transfer::set_data(fluid* aFluid, transfers* aTransfers)
{
	mFluid = aFluid;
	mTransfers = aTransfers;
	return *this;
}

void pbd::particle_transfer::apply(float aDeltaTime)
{
	auto& particleList         = mFluid->get<fluid::id::particle>();
	auto& transferSourceList   = mTransfers->hidden_list().get<hidden_transfers::id::source>();
	auto& transferTargetList   = mTransfers->hidden_list().get<hidden_transfers::id::target>();
	auto& transferTimeLeftList = mTransfers->hidden_list().get<hidden_transfers::id::time_left>();
	auto& transferringList     = particleList.hidden_list().get<pbd::hidden_particles::id::transferring>();
	auto& positionList         = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList      = particleList.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList           = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto& velocityList         = particleList.hidden_list().get<pbd::hidden_particles::id::velocity>();
	auto  deleteParticleList   = particles().share_hidden_data_from(particleList).request_length(mTransfers->hidden_list().requested_length());
	auto  deleteTransferList   = transfers().share_hidden_data_from(*mTransfers).request_length(mTransfers->hidden_list().requested_length());

	shader_provider::particle_transfer(particleList.index_buffer(), positionList.write().buffer(), radiusList.write().buffer(), inverseMassList.write().buffer(), velocityList.write().buffer(), transferSourceList.write().index_buffer(), transferTargetList.write().index_buffer(), transferTimeLeftList.write().buffer(), transferringList.write().buffer(), deleteParticleList.write().index_buffer(), deleteTransferList.write().index_buffer(), mTransfers->hidden_list().write().length(), deleteParticleList.write().length(), deleteTransferList.write().length(), aDeltaTime);

	deleteTransferList.delete_these();
	deleteParticleList.delete_these();
}
