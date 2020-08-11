#include "inter_particle_collision.h"

pbd::inter_particle_collision& pbd::inter_particle_collision::set_data(particles* aParticles, gpu_list<sizeof(uint32_t) * NEIGHBOR_LIST_MAX_LENGTH>* aNeighbors)
{
	mParticles = aParticles;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::inter_particle_collision::apply()
{
	auto& positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList = mParticles->hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();

	shader_provider::inter_particle_collision(mParticles->index_buffer(), positionList.write().buffer(), radiusList.buffer(), inverseMassList.buffer(), mNeighbors->buffer(), mParticles->length());
}
