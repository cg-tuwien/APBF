#include "box_collision.h"

pbd::box_collision& pbd::box_collision::add_particles(const particles& aParticles)
{
	mParticles += aParticles;
	return *this;
}

pbd::box_collision::boxes pbd::box_collision::add_box(const glm::vec3& aMin, const glm::vec3& aMax)
{
	mBoxes.request_length(16); // TODO
	mBoxes.hidden_list().request_length(16); // TODO
	auto result = mBoxes.increase_length(1);
	shader_provider::add_box(result.index_buffer(), mBoxes.hidden_list().write().buffer(), glm::vec4(aMin, 1), glm::vec4(aMax, 1));
	return result;
}

void pbd::box_collision::delete_boxes(boxes& aBox)
{
	aBox.delete_these();
}

void pbd::box_collision::apply()
{
	if (mParticles.empty()) return;

	auto& positionList = mParticles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& radiusList   = mParticles.hidden_list().get<pbd::hidden_particles::id::radius>();

	shader_provider::box_collision(mParticles.index_buffer(), positionList.write().buffer(), radiusList.buffer(), mBoxes.hidden_list().buffer(), mParticles.length(), mBoxes.length());
}
