#include "box_collision.h"

pbd::box_collision& pbd::box_collision::set_data(particles* aParticles, pbd::gpu_list<16>* aBoxMin, pbd::gpu_list<16>* aBoxMax)
{
	mParticles = aParticles;
	mBoxMin = aBoxMin;
	mBoxMax = aBoxMax;
	return *this;
}

void pbd::box_collision::apply()
{
	if (mParticles->empty()) return;

	auto& positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& radiusList   = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();

	shader_provider::box_collision(mParticles->index_buffer(), positionList.write().buffer(), radiusList.buffer(), mBoxMin->buffer(), mBoxMax->buffer(), mParticles->length(), mBoxMin->length());
}
