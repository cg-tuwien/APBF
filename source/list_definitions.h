#pragma once

#include "gpu_list.h"
#include "uninterleaved_list.h"
#include "indexed_list.h"

namespace pbd
{
	enum class hidden_particles_enum { position, velocity, inverse_mass, radius, pos_backup };
	using hidden_particles = uninterleaved_list<hidden_particles_enum, gpu_list<16>, gpu_list<16>, gpu_list<4>, gpu_list<4>, gpu_list<16>>;
	using particles = indexed_list<hidden_particles>;
	
	enum class fluid_enum { particle, target_radius, kernel_width, boundary_distance };
	using fluid = uninterleaved_list<fluid_enum, particles, gpu_list<4>, gpu_list<4>, gpu_list<4>>;

//	enum class NeighborhoodInfoEnum { cellStart, cellEnd };
//	using NeighborhoodInfo = GpuListUninterleaved<NeighborhoodInfoEnum, GpuList<GpuListType::Uint>, GpuList<GpuListType::Uint>>;
}
