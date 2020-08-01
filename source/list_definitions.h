#pragma once

#include "gpu_list.h"
#include "uninterleaved_list.h"
#include "indexed_list.h"

namespace pbd
{
	enum class hidden_particles_enum { position, velocity, mass, phase, radius, pos_backup };
	using hidden_particles = uninterleaved_list<hidden_particles_enum, gpu_list<12>, gpu_list<12>, gpu_list<4>, gpu_list<4>, gpu_list<4>, gpu_list<12>>;
	using particles = indexed_list<hidden_particles>;

//	enum class NeighborhoodInfoEnum { cellStart, cellEnd };
//	using NeighborhoodInfo = GpuListUninterleaved<NeighborhoodInfoEnum, GpuList<GpuListType::Uint>, GpuList<GpuListType::Uint>>;
}
