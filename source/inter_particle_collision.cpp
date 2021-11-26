#include "inter_particle_collision.h"

pbd::inter_particle_collision& pbd::inter_particle_collision::set_data(particles* aParticles, neighbors* aNeighbors)
{
	mParticles = aParticles;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::inter_particle_collision::apply()
{
	auto& positionList    = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList = mParticles->hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList      = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();
	auto  oldPositionList = positionList;

	shader_provider::inter_particle_collision(mParticles->index_buffer(), oldPositionList.buffer(), radiusList.buffer(), inverseMassList.buffer(), mNeighbors->buffer(), positionList.write().buffer(), mNeighbors->length());
}
