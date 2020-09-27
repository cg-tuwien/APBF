#pragma once

#include <gvk.hpp>
#include "gpu_list.h"

class user_controlled_boxes
{
public:
	user_controlled_boxes();
	void handle_input(const gvk::input_buffer& aInput);
	void add_box(const glm::vec3& aMin, const glm::vec3& aMax);
	void render(const glm::mat4& aViewProjection);

private:
	void update_buffer();

	bool mBufferOutdated;
	std::vector<glm::vec4> mBoxMinData;
	std::vector<glm::vec4> mBoxMaxData;
	avk::buffer mVertexBuffer;
	avk::buffer mIndexBuffer;
	pbd::gpu_list<16> mBoxMin;
	pbd::gpu_list<16> mBoxMax;
};
