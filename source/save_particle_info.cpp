#include "save_particle_info.h"
#include "../shaders/cpu_gpu_shared_config.h"

pbd::save_particle_info& pbd::save_particle_info::set_data(fluid* aFluid, neighbors* aNeighbors)
{
	mFluid     = aFluid;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::save_particle_info::apply()
{
	auto&  kernelWidthList = mFluid->get<pbd::fluid::id::kernel_width>();
	auto& targetRadiusList = mFluid->get<pbd::fluid::id::target_radius>();
	auto& boundaryDistList = mFluid->get<pbd::fluid::id::boundary_distance>();
	auto&        particles = mFluid->get<pbd::fluid::id::particle>();
//	auto&     velocityList = particles.hidden_list().get<pbd::hidden_particles::id::velocity>();
	auto&     positionList = particles.hidden_list().get<pbd::hidden_particles::id::position>();
//	auto&    posBackupList = particles.hidden_list().get<pbd::hidden_particles::id::pos_backup>();
	auto&       radiusList = particles.hidden_list().get<pbd::hidden_particles::id::radius>();

	auto      indices =  particles.index_read();
	auto    positions =     positionList.read<glm::ivec4>();
	auto        radii =       radiusList.read<     float>();
	auto  kernelWidth =  kernelWidthList.read<     float>();
	auto targetRadius = targetRadiusList.read<     float>();
	auto boundaryDist = boundaryDistList.read<glm:: uint>();
	auto     nbrPairs =      mNeighbors->read<glm::uvec2>();
	auto     nbrCount = std::vector<unsigned int>();
	auto       radius = std::vector<float>();
	auto   centerDist = std::vector<float>();
	auto    centerPos = glm::vec3(0, 10, -60);

	nbrCount  .resize(indices.size());
	radius    .resize(indices.size());
	centerDist.resize(indices.size());

	for (auto& count : nbrCount) {
		count = 0;
	}

	for (auto& pair : nbrPairs) {
		nbrCount[pair.x]++;
	}
	
	for (auto i = 0u; i < indices.size(); i++) {
		auto id = indices[i];
		auto pos = glm::vec3(positions[id]) / static_cast<float>(POS_RESOLUTION);

		centerDist[i] = glm::distance(pos, centerPos);
		radius    [i] = radii[id];
	}

	// write files

	{
		auto toFile = std::ofstream("centerDist.txt");
		for (auto& dist : centerDist) toFile << dist << ";";
	}

	{
		auto toFile = std::ofstream("radius.txt");
		for (auto& data : radius) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream("neighborCount.txt");
		for (auto& data : nbrCount) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream("kernelWidth.txt");
		for (auto& data : kernelWidth) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream("targetRadius.txt");
		for (auto& data : targetRadius) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream("boundaryDistance.txt");
		for (auto& data : boundaryDist) toFile << (data / POS_RESOLUTION) << ";";
	}
}
