#include "user_controlled_boxes.h"
#include "shader_provider.h"
#include "../shaders/cpu_gpu_shared_config.h"

user_controlled_boxes::user_controlled_boxes()
{
	auto vertices = std::vector<glm::vec3>({{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}});
	auto indices  = std::vector<uint16_t>({0, 1, 2,  0, 2, 3,  0, 5, 1,  1, 5, 6,  1, 6, 7,  1, 7, 2,  2, 7, 3,  3, 7, 4,  0, 3, 4,  0, 4, 5,  4, 6, 5,  4, 7, 6});
	mBufferOutdated = true;
	mVertexBuffer = gvk::context().create_buffer(avk::memory_usage::device, vk::BufferUsageFlagBits::eVertexBuffer, avk::vertex_buffer_meta::create_from_data(vertices));
	mIndexBuffer  = gvk::context().create_buffer(avk::memory_usage::device, vk::BufferUsageFlagBits:: eIndexBuffer, avk:: index_buffer_meta::create_from_data(indices));
	mVertexBuffer->fill(vertices.data(), 0, avk::sync::wait_idle(true));
	mIndexBuffer ->fill(indices .data(), 0, avk::sync::wait_idle(true));
	mSelectedIdx = 0;
}

void user_controlled_boxes::handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos)
{
	if (gvk::input().mouse_button_pressed(0)) {
		update_cursor_data(aInverseViewProjection, aCameraPos);
		auto vsClickDist = std::numeric_limits<float>().infinity();
		mLockedAxis = 0;
		mSelectedIdx = -1;
		for (auto i = 0; i < mBoxMinData.size(); i++) {
			float d;
			d = (mBoxMinData[i].x - aCameraPos.x) / mCursorDirVS.x; if (d < vsClickDist && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 0; mSelectedIdx = i; }
			d = (mBoxMinData[i].y - aCameraPos.y) / mCursorDirVS.y; if (d < vsClickDist && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 1; mSelectedIdx = i; }
			d = (mBoxMinData[i].z - aCameraPos.z) / mCursorDirVS.z; if (d < vsClickDist && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 2; mSelectedIdx = i; }
			d = (mBoxMaxData[i].x - aCameraPos.x) / mCursorDirVS.x; if (d < vsClickDist && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 0; mSelectedIdx = i; }
			d = (mBoxMaxData[i].y - aCameraPos.y) / mCursorDirVS.y; if (d < vsClickDist && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 1; mSelectedIdx = i; }
			d = (mBoxMaxData[i].z - aCameraPos.z) / mCursorDirVS.z; if (d < vsClickDist && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 2; mSelectedIdx = i; }
		}
		mCursorPosVS = aCameraPos + mCursorDirVS * vsClickDist;
#if DIMENSIONS < 3
		mLockedAxis = 2;
#endif
	}

	else if (gvk::input().mouse_button_down(0) && mSelectedIdx >= 0) {
		update_cursor_data(aInverseViewProjection, aCameraPos);
		auto newPos = cursor_pos_on_plane(mCursorPosVS, mLockedAxis);
		auto shift = glm::vec4(newPos - mCursorPosVS, 0.0f);
		auto test = glm::length(shift) > 1;
		mBoxMinData[mSelectedIdx] += shift;
		mBoxMaxData[mSelectedIdx] += shift;
		mCursorPosVS = newPos;
		mBufferOutdated = true;
	}
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
	shader_provider::render_boxes(mVertexBuffer, mIndexBuffer, mBoxMin.buffer(), mBoxMax.buffer(), aViewProjection, mBoxMinData.size(), mSelectedIdx);
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

void user_controlled_boxes::update_cursor_data(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos)
{
	auto ndc = gvk::input().cursor_position() / glm::dvec2(gvk::context().main_window()->resolution()) * 2.0 - 1.0;
	auto vsP = aInverseViewProjection * glm::vec4(ndc, 1.0f, 1.0f);
	auto vsCursorPoint = glm::vec3(vsP) / vsP.w;
	mCursorDirVS = glm::normalize(vsCursorPoint - aCameraPos);
	mCameraPosVS = aCameraPos;
}

glm::vec3 user_controlled_boxes::cursor_pos_on_plane(const glm::vec3& aPlanePoint, int aAxis)
{
	auto dist = (aPlanePoint[aAxis] - mCameraPosVS[aAxis]) / mCursorDirVS[aAxis];
	auto result = mCameraPosVS + mCursorDirVS * dist;
	result[aAxis] = aPlanePoint[aAxis]; // prevent drifting caused by rounding errors
	return result;
}

bool user_controlled_boxes::in_box(const glm::vec3& aPos, const glm::vec3& aMin, const glm::vec3& aMax)
{
	return glm::all(glm::greaterThanEqual(aPos, aMin)) && glm::all(glm::lessThanEqual(aPos, aMax));
}
