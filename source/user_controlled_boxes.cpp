#include "user_controlled_boxes.h"
#include "shader_provider.h"

user_controlled_boxes::user_controlled_boxes()
{
	auto vertices = std::vector<glm::vec3>({{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}});
	auto indices  = std::vector<uint16_t>({0, 1, 2,  0, 2, 3,  0, 5, 1,  1, 5, 6,  1, 6, 7,  1, 7, 2,  2, 7, 3,  3, 7, 4,  0, 3, 4,  0, 4, 5,  4, 6, 5,  4, 7, 6});
	mBufferOutdated = true;
	mVertexBuffer = gvk::context().create_buffer(avk::memory_usage::device, vk::BufferUsageFlagBits::eVertexBuffer, avk::vertex_buffer_meta::create_from_data(vertices));
	mIndexBuffer  = gvk::context().create_buffer(avk::memory_usage::device, vk::BufferUsageFlagBits:: eIndexBuffer, avk:: index_buffer_meta::create_from_data(indices));
	mVertexBuffer->fill(vertices.data(), 0, avk::sync::wait_idle(true));
	mIndexBuffer ->fill(indices .data(), 0, avk::sync::wait_idle(true));
}

void user_controlled_boxes::handle_input(const gvk::input_buffer& aInput)
{
	// TODO
}

void user_controlled_boxes::add_box(const glm::vec3& aMin, const glm::vec3& aMax)
{
	mBoxMinData.push_back(glm::vec4(aMin, 1.0f));
	mBoxMaxData.push_back(glm::vec4(aMax, 1.0f));
	mBufferOutdated = true;
}

void user_controlled_boxes::render(const glm::mat4& aViewProjection)
{
	update_buffer();
	shader_provider::render_boxes(mVertexBuffer, mIndexBuffer, mBoxMin.buffer(), mBoxMax.buffer(), aViewProjection, mBoxMinData.size());
}

void user_controlled_boxes::update_buffer()
{
	if (!mBufferOutdated) return;
	mBoxMin.request_length(std::max(1ui64, mBoxMinData.size()));
	mBoxMax.request_length(std::max(1ui64, mBoxMaxData.size()));
	pbd::algorithms::copy_bytes(mBoxMinData.data(), mBoxMin.write().buffer(), mBoxMinData.size() * sizeof(glm::vec4));
	pbd::algorithms::copy_bytes(mBoxMaxData.data(), mBoxMax.write().buffer(), mBoxMaxData.size() * sizeof(glm::vec4));
	mBufferOutdated = false;
}
