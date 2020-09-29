#pragma once

#include <gvk.hpp>
#include "gpu_list.h"

class user_controlled_boxes
{
public:
	user_controlled_boxes();
	void handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos);
	void add_box(const glm::vec3& aMin, const glm::vec3& aMax);
	void render(const glm::mat4& aViewProjection);
	pbd::gpu_list<16>& box_min();
	pbd::gpu_list<16>& box_max();

private:
	void update_buffer();
	void update_cursor_data(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos);
	int index_of_clicked_box(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos);
	glm::vec3 cursor_pos_on_plane(const glm::vec3& aPlanePoint, int aAxis);
	bool in_box(const glm::vec3& aPos, const glm::vec3& aMin, const glm::vec3& aMax);

	bool mBufferOutdated;
	std::vector<glm::vec4> mBoxMinData;
	std::vector<glm::vec4> mBoxMaxData;
	std::vector<glm::vec4> mBoxMinDataFloating; // visible, but not added as a real box yet
	std::vector<glm::vec4> mBoxMaxDataFloating; // e.g. boxes that currently get duplicated
	std::vector<bool> mSelected;
	avk::buffer mVertexBuffer;
	avk::buffer mIndexBuffer;
	pbd::gpu_list<16> mBoxMin;
	pbd::gpu_list<16> mBoxMax;
	pbd::gpu_list<4> mBoxSelected;
	bool mDragging;
	int mLockedAxis;
	glm::vec3 mCursorPosVS;

	glm::vec3 mCursorDirVS;
	glm::vec3 mCameraPosVS;
};
