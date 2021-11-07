#pragma once

#include "gpu_list.h"
#include "uninterleaved_list.h"
#include "indexed_list.h"

namespace pbd
{
	enum class hidden_particles_enum { position, radius };
	using hidden_particles = uninterleaved_list<hidden_particles_enum, gpu_list<sizeof(glm::ivec4)>, gpu_list<sizeof(float)>>;
	using particles = indexed_list<hidden_particles>;

	using neighbors = gpu_list<sizeof(glm::uvec2)>;
}
