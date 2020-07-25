#pragma once

#include "gpu_list.h"
#include "GpuListUninterleaved.h"
#include "IndexedList.h"

namespace pbd
{
	enum class HiddenParticlesEnum { position, velocity, mass, phase, radius, posBackup };
	using HiddenParticles = GpuListUninterleaved<HiddenParticlesEnum, GpuList<GpuListType::Int3>, GpuList<GpuListType::Float3>, GpuList<GpuListType::Float>, GpuList<GpuListType::Uint>, GpuList<GpuListType::Float>, GpuList<GpuListType::Int3>>;
	using Particles = IndexedList<HiddenParticles>;

	enum class NeighborhoodInfoEnum { cellStart, cellEnd };
	using NeighborhoodInfo = GpuListUninterleaved<NeighborhoodInfoEnum, GpuList<GpuListType::Uint>, GpuList<GpuListType::Uint>>;
}
