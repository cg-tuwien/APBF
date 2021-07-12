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
	auto&     positionList = particles.hidden_list().get<pbd::hidden_particles::id::position>();
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

void pbd::save_particle_info::save_as_svg(uint32_t aSvgId, const glm::vec2& aViewBoxMin, const glm::vec2& aViewBoxMax, float aRenderScale)
{
	auto& boundarinessList = mFluid->get<pbd::fluid::id::boundariness>();
	auto&        particles = mFluid->get<pbd::fluid::id::particle>();
	auto&     positionList = particles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto&       radiusList = particles.hidden_list().get<pbd::hidden_particles::id::radius>();

	auto      indices =  particles.index_read();
	auto    positions =     positionList.read<glm::ivec4>();
	auto        radii =       radiusList.read<     float>();
	auto boundariness = boundarinessList.read<     float>();

	auto svg = std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"particle\" style=\"fill:#0000ff;fill-rule:evenodd;stroke-width:1\" />", aRenderScale);
	svg += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"boundaryParticle\" style=\"fill:#ff0000;fill-rule:evenodd;stroke-width:1\" />", aRenderScale);

	for (auto i = 0u; i < indices.size(); i++) {
		auto id = indices[i];
		auto pos = glm::vec2(positions[id]) / static_cast<float>(POS_RESOLUTION);
		auto rad = radii[id];
		auto bdr = boundariness[i] >= 1.0f;
		pos.y = -pos.y;

		auto matrix = std::format("matrix({},0,0,{},{},{})", rad, rad, pos.x, pos.y);
		svg += std::format("<use transform=\"{}\" xlink:href=\"{}\" />", matrix, bdr ? "#boundaryParticle" : "#particle");
	}

	svg = std::format("<svg viewBox=\"{} {} {} {}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:svg=\"http://www.w3.org/2000/svg\">{}</svg>", aViewBoxMin.x, -aViewBoxMax.y, aViewBoxMax.x - aViewBoxMin.x, aViewBoxMax.y - aViewBoxMin.y, svg);

	{
		auto toFile = std::ofstream(std::format("particles_{}.svg", aSvgId));
		toFile << svg;
	}
}
