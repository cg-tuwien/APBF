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
	mDragging = false;
}

void user_controlled_boxes::handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos)
{
	if (gvk::input().mouse_button_pressed(0) || gvk::input().mouse_button_pressed(1)) {
		auto selectedIdx = index_of_clicked_box(aInverseViewProjection, aCameraPos, !gvk::input().key_down(gvk::key_code::left_alt));
		if (gvk::input().key_down(gvk::key_code::left_shift) || gvk::input().key_down(gvk::key_code::left_shift)) {
			if (selectedIdx >= 0) mSelected[selectedIdx] = !mSelected[selectedIdx];
			mBufferOutdated = true;
		}
		else if (selectedIdx >= 0 && mSelected[selectedIdx]) {
			mDragging = true;
			if (gvk::input().mouse_button_pressed(1)) {
				for (auto i = 0; i < mSelected.size(); i++) if (mSelected[i]) {
					mBoxMinDataFloating.push_back(mBoxMinData[i]);
					mBoxMaxDataFloating.push_back(mBoxMaxData[i]);
				}
			}
		}
		else {
			std::fill(mSelected.begin(), mSelected.end(), false);
			if (selectedIdx >= 0) mSelected[selectedIdx] = true;
			mBufferOutdated = true;
		}
	}

	else if ((gvk::input().mouse_button_down(0) || gvk::input().mouse_button_down(1)) && mDragging) {
		update_cursor_data(aInverseViewProjection, aCameraPos);
		auto newPos = cursor_pos_on_plane(mCursorPosVS, mLockedAxis);
		auto shift = glm::vec4(newPos - mCursorPosVS, 0.0f);
		if (gvk::input().mouse_button_down(0)) {
			for (auto i = 0; i < mSelected.size(); i++) if (mSelected[i]) {
				mBoxMinData[i] += shift;
				mBoxMaxData[i] += shift;
			}
		}
		if (gvk::input().mouse_button_down(1)) {
			for (auto i = 0; i < mBoxMinDataFloating.size(); i++) {
				mBoxMinDataFloating[i] += shift;
				mBoxMaxDataFloating[i] += shift;
			}
		}
		mCursorPosVS = newPos;
		mBufferOutdated = true;
	}

	if (gvk::input().mouse_button_released(1) && mDragging) {
		mBoxMinData.insert(mBoxMinData.end(), mBoxMinDataFloating.begin(), mBoxMinDataFloating.end());
		mBoxMaxData.insert(mBoxMaxData.end(), mBoxMaxDataFloating.begin(), mBoxMaxDataFloating.end());
		mSelected.insert(mSelected.end(), mBoxMinDataFloating.size(), false);
		mBoxMinDataFloating.clear();
		mBoxMaxDataFloating.clear();
		mBufferOutdated = true;
	}

	if (gvk::input().key_pressed(gvk::key_code::del)) {
		for (auto i = static_cast<int>(mSelected.size()) - 1; i >= 0; i--) if (mSelected[i]) {
			mBoxMinData.erase(mBoxMinData.begin() + i);
			mBoxMaxData.erase(mBoxMaxData.begin() + i);
			mSelected.erase(mSelected.begin() + i);
			mBufferOutdated = true;
		}
	}

	// x, y and z keys for locking x, y or z axis
	if (std::find(gvk::input().entered_characters().begin(), gvk::input().entered_characters().end(),  88u) != gvk::input().entered_characters().end()) mLockedAxis = 0; // X
	if (std::find(gvk::input().entered_characters().begin(), gvk::input().entered_characters().end(), 120u) != gvk::input().entered_characters().end()) mLockedAxis = 0; // x
	if (std::find(gvk::input().entered_characters().begin(), gvk::input().entered_characters().end(),  89u) != gvk::input().entered_characters().end()) mLockedAxis = 1; // Y
	if (std::find(gvk::input().entered_characters().begin(), gvk::input().entered_characters().end(), 121u) != gvk::input().entered_characters().end()) mLockedAxis = 1; // y
	if (std::find(gvk::input().entered_characters().begin(), gvk::input().entered_characters().end(),  90u) != gvk::input().entered_characters().end()) mLockedAxis = 2; // Z
	if (std::find(gvk::input().entered_characters().begin(), gvk::input().entered_characters().end(), 122u) != gvk::input().entered_characters().end()) mLockedAxis = 2; // z

#if DIMENSIONS < 3
	mLockedAxis = 2;
#endif
}

void user_controlled_boxes::add_box(const glm::vec3& aMin, const glm::vec3& aMax)
{
	mBoxMinData.push_back(glm::vec4(aMin, 1.0f));
	mBoxMaxData.push_back(glm::vec4(aMax, 1.0f));
	mSelected.push_back(false);
	mBufferOutdated = true;
}

void user_controlled_boxes::render(const glm::mat4& aViewProjection)
{
	update_buffer();
	shader_provider::render_boxes(mVertexBuffer, mIndexBuffer, mBoxMin.buffer(), mBoxMax.buffer(), mBoxSelected.buffer(), aViewProjection, static_cast<uint32_t>(mBoxMinData.size() + mBoxMinDataFloating.size()));
}

pbd::gpu_list<16>& user_controlled_boxes::box_min()
{
	update_buffer();
	return mBoxMin;
}

pbd::gpu_list<16>& user_controlled_boxes::box_max()
{
	update_buffer();
	return mBoxMax;
}

void user_controlled_boxes::update_buffer()
{
	if (!mBufferOutdated) return;
	auto mSelectedData = std::vector<vk::Bool32>(mSelected.begin(), mSelected.end());
	auto totalBoxCount = mBoxMinData.size() + mBoxMinDataFloating.size();
	mBoxMin.request_length(std::max(1ui64, totalBoxCount)).set_length(mBoxMinData.size()); // exploiting the fact that the box collision constraint uses the length value,
	mBoxMax.request_length(std::max(1ui64, totalBoxCount)).set_length(mBoxMaxData.size()); // therefore the floating boxes won't have a collision
	mBoxSelected.request_length(std::max(1ui64, totalBoxCount));
	pbd::algorithms::copy_bytes(mSelectedData.data(), mBoxSelected.write().buffer(), mSelected.size() * sizeof(vk::Bool32));
	pbd::algorithms::copy_bytes(mBoxMinData.data(), mBoxMin.write().buffer(), mBoxMinData.size() * sizeof(glm::vec4));
	pbd::algorithms::copy_bytes(mBoxMaxData.data(), mBoxMax.write().buffer(), mBoxMaxData.size() * sizeof(glm::vec4));
	pbd::algorithms::copy_bytes(mBoxMinDataFloating.data(), mBoxMin.write().buffer(), mBoxMinDataFloating.size() * sizeof(glm::vec4), 0, mBoxMinData.size() * sizeof(glm::vec4));
	pbd::algorithms::copy_bytes(mBoxMaxDataFloating.data(), mBoxMax.write().buffer(), mBoxMaxDataFloating.size() * sizeof(glm::vec4), 0, mBoxMaxData.size() * sizeof(glm::vec4));
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

int user_controlled_boxes::index_of_clicked_box(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos, bool aFrontmost)
{
	mLockedAxis = 0;
	mDragging = false;
	update_cursor_data(aInverseViewProjection, aCameraPos);
	auto vsClickDist = aFrontmost ? std::numeric_limits<float>().infinity() : 0.0f;
	auto selectedIdx = -1;
	for (auto i = 0; i < mBoxMinData.size(); i++) {
		float d;
		d = (mBoxMinData[i].x - aCameraPos.x) / mCursorDirVS.x; if ((d < vsClickDist == aFrontmost) && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 0; selectedIdx = i; }
		d = (mBoxMinData[i].y - aCameraPos.y) / mCursorDirVS.y; if ((d < vsClickDist == aFrontmost) && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 1; selectedIdx = i; }
		d = (mBoxMinData[i].z - aCameraPos.z) / mCursorDirVS.z; if ((d < vsClickDist == aFrontmost) && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 2; selectedIdx = i; }
		d = (mBoxMaxData[i].x - aCameraPos.x) / mCursorDirVS.x; if ((d < vsClickDist == aFrontmost) && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 0; selectedIdx = i; }
		d = (mBoxMaxData[i].y - aCameraPos.y) / mCursorDirVS.y; if ((d < vsClickDist == aFrontmost) && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 1; selectedIdx = i; }
		d = (mBoxMaxData[i].z - aCameraPos.z) / mCursorDirVS.z; if ((d < vsClickDist == aFrontmost) && in_box(aCameraPos + mCursorDirVS * d * 1.0001f, mBoxMinData[i], mBoxMaxData[i])) { vsClickDist = d; mLockedAxis = 2; selectedIdx = i; }
	}
	mCursorPosVS = aCameraPos + mCursorDirVS * vsClickDist;
	return selectedIdx;
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
