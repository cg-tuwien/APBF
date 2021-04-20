#include "glue.h"

pbd::glue& pbd::glue::set_data(particles* aParticles, neighbors* aNeighbors, glue_list* aGlue)
{
	mParticles = aParticles;
	mNeighbors = aNeighbors;
	mGlue      = aGlue;
	assert(aGlue->get<glue_list::id::particle_0>().hidden_list() == aGlue->get<glue_list::id::particle_1>().hidden_list());
	return *this;
}

pbd::glue& pbd::glue::set_stability(float aStability)
{
	mStability = aStability;
	return *this;
}

pbd::glue& pbd::glue::set_elasticity(float aElasticity)
{
	mElasticity = aElasticity;
	return *this;
}

void pbd::glue::apply()
{
	auto& glueIndex0      = mGlue->get<glue_list::id::particle_0>();
	auto& glueIndex1      = mGlue->get<glue_list::id::particle_1>();
	auto& glueDistance    = mGlue->get<glue_list::id::distance>();
	auto& positionList    = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList = mParticles->hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList      = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();

	shader_provider::glue(glueIndex0.index_buffer(), glueIndex1.index_buffer(), glueDistance.buffer(), positionList.write().buffer(), inverseMassList.buffer(), mGlue->length(), mStability, mElasticity);
}

void pbd::glue::applyNewGlue()
{
	auto& glueIndex0   = mGlue->get<glue_list::id::particle_0>();
	auto& glueIndex1   = mGlue->get<glue_list::id::particle_1>();
	auto& glueDistance = mGlue->get<glue_list::id::distance>();
	auto& positionList = mParticles->hidden_list().get<pbd::hidden_particles::id::position>();
	auto& radiusList   = mParticles->hidden_list().get<pbd::hidden_particles::id::radius>();

	mGlue->set_length(shader_provider::generate_glue(mParticles->index_buffer(), mNeighbors->buffer(), positionList.buffer(), radiusList.buffer(), glueIndex0.write().index_buffer(), glueIndex1.write().index_buffer(), glueDistance.write().buffer(), mNeighbors->length()));
}
