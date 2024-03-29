#include "save_particle_info.h"
#include "settings.h"
#include "measurements.h"
#include "../shaders/cpu_gpu_shared_config.h"

pbd::save_particle_info& pbd::save_particle_info::set_data(fluid* aFluid, neighbors* aNeighbors, transfers* aTransfers)
{
	mFluid     = aFluid;
	mNeighbors = aNeighbors;
	mTransfers = aTransfers;
	return *this;
}

pbd::save_particle_info& pbd::save_particle_info::set_boxes(const pbd::gpu_list<16>* aBoxMin, const pbd::gpu_list<16>* aBoxMax)
{
	mBoxMin = aBoxMin;
	mBoxMax = aBoxMax;
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
	auto&  inverseMassList = particles.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();

	auto      indices =  particles.index_read();
	auto    positions =     positionList.read<glm::ivec4>();
	auto        radii =       radiusList.read<     float>();
	auto  kernelWidth =  kernelWidthList.read<     float>();
	auto targetRadius = targetRadiusList.read<     float>();
	auto    invMasses =  inverseMassList.read<     float>();
	auto boundaryDist = boundaryDistList.read<glm:: uint>();
	auto     nbrPairs =      mNeighbors->read<glm::uvec2>();
	auto     nbrCount = std::vector<unsigned int>();
	auto    sortedIdx = std::vector<unsigned int>();
	auto       radius = std::vector<float>();
	auto  inverseMass = std::vector<float>();
	auto   centerDist = std::vector<float>();
	auto      bdrDist = std::vector<float>();
	auto     position = std::vector<glm::vec3>();
	auto    centerPos = glm::vec3(0, 10, -60);

	nbrCount   .resize(indices.size());
	sortedIdx  .resize(indices.size());
	radius     .resize(indices.size());
	inverseMass.resize(indices.size());
	centerDist .resize(indices.size());
	bdrDist    .resize(indices.size());
	position   .resize(indices.size());

	for (auto& count : nbrCount) {
		count = 0;
	}

	for (auto& pair : nbrPairs) {
		nbrCount[pair.x]++;
	}
	
	for (auto i = 0u; i < indices.size(); i++) {
		auto id = indices[i];
		auto pos = glm::vec3(positions[id]) / static_cast<float>(POS_RESOLUTION);

		bdrDist    [i] = boundaryDist[i] / static_cast<float>(POS_RESOLUTION);
		centerDist [i] = glm::distance(pos, centerPos);
		radius     [i] = radii[id];
		inverseMass[i] = invMasses[id];
		position   [i] = pos;
		sortedIdx  [i] = i;
	}

	// write files

	std::filesystem::create_directories(PARTICLE_INFO_FOLDER_NAME);
	{
		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/centerDist.txt");
		for (auto& dist : centerDist) toFile << dist << ";";
	}

	{
		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/radius.txt");
		for (auto& data : radius) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/neighborCount.txt");
		for (auto& data : nbrCount) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/kernelWidth.txt");
		for (auto& data : kernelWidth) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/targetRadius.txt");
		for (auto& data : targetRadius) toFile << data << ";";
	}

	{
		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/boundaryDistance.txt");
		for (auto& data : bdrDist) toFile << data << ";";
	}

	{
		std::sort(sortedIdx.begin(), sortedIdx.end(), [&](unsigned int a, unsigned int b) { return centerDist[a] < centerDist[b]; });

		auto toFile = std::ofstream(PARTICLE_INFO_FOLDER_NAME "/data.csv");
		toFile << "center distance,boundary distance,kernel width,neighbor count,radius,target radius,inverse mass,x,y,z" << std::endl;
		for (auto i = 0u; i < indices.size(); i++) {
			auto idx = sortedIdx[i];
			toFile << centerDist  [idx] << ","
			       << bdrDist     [idx] << ","
			       << kernelWidth [idx] << ","
			       << nbrCount    [idx] << ","
			       << radius      [idx] << ","
			       << targetRadius[idx] << ","
			       << inverseMass [idx] << ","
			       << position    [idx].x << ","
			       << position    [idx].y << ","
			       << position    [idx].z << std::endl;
		}
	}
}

void pbd::save_particle_info::save_as_svg(uint32_t aSvgId, const glm::vec2& aViewBoxMin, const glm::vec2& aViewBoxMax, float aRenderScale, float aMaxExpectedBoundaryDist)
{
	auto includeBoxes   = mBoxMin != nullptr && mBoxMax != nullptr && pbd::settings::renderBoxes;
	auto includeKernels = pbd::settings::color == 3;
	auto colorGradient  = false;
	auto colorCount     = 0u;
	auto strokeWidth    = glm::compMin(aViewBoxMax - aViewBoxMin) / 1000.0f;

	switch (pbd::settings::color) {
		case  1: colorGradient =  true; colorCount =  0u; break;
		case  2: colorGradient = false; colorCount = 10u; break;
		default: colorGradient = false; colorCount =  2u; break;
	}

	auto& boundarinessList = mFluid->get<pbd::fluid::id::boundariness>();
	auto& boundaryDistList = mFluid->get<pbd::fluid::id::boundary_distance>();
	auto&  kernelWidthList = mFluid->get<pbd::fluid::id::kernel_width>();
	auto&        particles = mFluid->get<pbd::fluid::id::particle>();
	auto&     positionList = particles.hidden_list().get<pbd::hidden_particles::id::position>();
	auto&       radiusList = particles.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto&  transferSrcList = mTransfers->hidden_list().get<hidden_transfers::id::source>();
	auto&  transferTgtList = mTransfers->hidden_list().get<hidden_transfers::id::target>();

	auto      indices =       particles.index_read();
	auto transfersSrc = transferSrcList.index_read();
	auto transfersTgt = transferTgtList.index_read();
	auto    positions =     positionList.read<glm::ivec4>();
	auto        radii =       radiusList.read<     float>();
	auto boundariness = boundarinessList.read<     float>();
	auto boundaryDist = boundaryDistList.read<glm:: uint>();
	auto  kernelWidth =  kernelWidthList.read<     float>();

	auto particleColor = std::vector<unsigned int>();
	particleColor.resize(positions.size());

	if (pbd::settings::color == 2) {
		for (auto i = 0u; i < particleColor.size(); i++) particleColor[i] = 0u;
		for (auto i = 0u; i < transfersSrc.size(); i++) {
			particleColor[transfersSrc[i]] = (i % (colorCount - 1u)) + 1u;
			particleColor[transfersTgt[i]] = (i % (colorCount - 1u)) + 1u;
		}
	}
	else if (pbd::settings::color != 1) {
		for (auto i = 0u; i < particleColor.size(); i++) particleColor[indices[i]] = boundariness[i] >= 1.0f;
	}

	{
		auto neighborPairCount = 0u;
		mNeighbors->length()->read(&neighborPairCount, 0, avk::sync::wait_idle(true));

		std::filesystem::create_directories(SVG_FOLDER_NAME);
		save_additional_info(aSvgId, indices.size(), neighborPairCount);
	}
	if (DIMENSIONS > 2) return;

	auto svg          = std::string();
	auto svgOriginals = std::string();
	auto svgParticles = std::string();
	auto svgKernels   = std::string();

	if (colorCount > 0) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p0\" style=\"fill:#0000ff;stroke-width:1\" />", aRenderScale);
	if (colorCount > 1) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p1\" style=\"fill:#ff0000;stroke-width:1\" />", aRenderScale);
	if (colorCount > 2) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p2\" style=\"fill:#00ff00;stroke-width:1\" />", aRenderScale);
	if (colorCount > 3) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p3\" style=\"fill:#ff8000;stroke-width:1\" />", aRenderScale);
	if (colorCount > 4) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p4\" style=\"fill:#ffff00;stroke-width:1\" />", aRenderScale);
	if (colorCount > 5) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p5\" style=\"fill:#ff00ff;stroke-width:1\" />", aRenderScale);
	if (colorCount > 6) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p6\" style=\"fill:#00ffff;stroke-width:1\" />", aRenderScale);
	if (colorCount > 7) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p7\" style=\"fill:#008000;stroke-width:1\" />", aRenderScale);
	if (colorCount > 8) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p8\" style=\"fill:#800000;stroke-width:1\" />", aRenderScale);
	if (colorCount > 9) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"{}\" id=\"p9\" style=\"fill:#ffaec8;stroke-width:1\" />", aRenderScale);
	if (includeKernels) svgOriginals += std::format("<circle cx=\"0\" cy=\"0\" r=\"1\" id=\"k\" style=\"fill:none;stroke:#ff7f00;stroke-width:{};vector-effect:non-scaling-stroke\" />", strokeWidth);
	if (includeBoxes  ) svgOriginals += "<rect x=\"0\" y=\"0\" width=\"1\" height=\"1\" id=\"box\" style=\"fill:#bffeff;stroke-width:1\" />";

	for (auto i = 0u; i < indices.size(); i++) {
		auto id = indices[i];
		auto pos = glm::vec2(positions[id]) / static_cast<float>(POS_RESOLUTION);
		auto rad = radii[id];
		auto ker = kernelWidth[i];
		pos.y = -pos.y;

		if (colorGradient) {
			// copied from main.cpp and instanced.vert
			auto color1 = glm::vec3(0, 0, 1);
			auto color2 = glm::vec3(0.62, 0.96, 0.83);
			auto color1Float = 0.0f;
			auto color2Float = aMaxExpectedBoundaryDist * POS_RESOLUTION * 0.8f;
			auto a = glm::clamp((boundaryDist[i] - color1Float) / (color2Float - color1Float), 0.0f, 1.0f);
			auto col = glm::uvec3(glm::mix(color1, color2, a) * 255.0f);
			svgParticles += std::format("<circle cx=\"{}\" cy=\"{}\" r=\"{}\" style=\"fill:#{:02x}{:02x}{:02x};stroke-width:1\" />", pos.x, pos.y, rad * aRenderScale, col.r, col.g, col.b);
		} else {
			auto color = particleColor[id];
			auto matrix = std::format("matrix({},0,0,{},{},{})", rad, rad, pos.x, pos.y);
			svgParticles += std::format("<use transform=\"{}\" xlink:href=\"#p{}\" />", matrix, color);
		}

		if (includeKernels) {
			auto matrix = std::format("matrix({},0,0,{},{},{})", ker, ker, pos.x, pos.y);
			svgKernels += std::format("<use transform=\"{}\" xlink:href=\"#k\" />", matrix);
		}
	}

	if (!svgOriginals.empty()) svg += std::format("<g id=\"originals\" style=\"display:none\">{}</g>", svgOriginals);
	if (includeBoxes         ) svg += boxes_to_svg();
	if (includeKernels       ) svg += std::format("<g id=\"kernels\">{}</g>", svgKernels);
	svg += std::format("<g id=\"particles\">{}</g>", svgParticles);
	svg = std::format("<svg viewBox=\"{} {} {} {}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:svg=\"http://www.w3.org/2000/svg\">{}</svg>", aViewBoxMin.x, -aViewBoxMax.y, aViewBoxMax.x - aViewBoxMin.x, aViewBoxMax.y - aViewBoxMin.y, svg);

	{
		auto toFile = std::ofstream(std::format(SVG_FOLDER_NAME "/particles_{}.svg", aSvgId));
		toFile << svg;
	}

//	{
//		auto toFile = std::ofstream(SVG_FOLDER_NAME "/particleCount.txt", aSvgId == 0u ? std::ios_base::out : std::ios_base::app);
//		toFile << indices.size() << std::endl;
//	}
//
//	{
//		auto toFile = std::ofstream(SVG_FOLDER_NAME "/neighborPairCount.txt", aSvgId == 0u ? std::ios_base::out : std::ios_base::app);
//		toFile << neighborPairCount << std::endl;
//	}
}

std::string pbd::save_particle_info::boxes_to_svg()
{
	auto svg = std::string();

	auto boxMin = mBoxMin->read<glm::vec4>();
	auto boxMax = mBoxMax->read<glm::vec4>();
	assert(boxMin.size() == boxMax.size());

	for (auto i = 0u; i < boxMin.size(); i++) {
		auto bMin = glm::vec2(boxMin[i]);
		auto bMax = glm::vec2(boxMax[i]);
		auto size = bMax - bMin;
		auto matrix = std::format("matrix({},0,0,{},{},{})", size.x, size.y, bMin.x, -bMax.y);
		svg += std::format("<use transform=\"{}\" xlink:href=\"#box\" />", matrix);
	}

	return std::format("<g id=\"boxes\">{}</g>", svg);
}

void pbd::save_particle_info::save_additional_info(uint32_t aSvgId, size_t aParticleCount, size_t aNeighborPairCount)
{
	auto toFile = std::ofstream(std::format(SVG_FOLDER_NAME "/measurements_{}.txt", aSvgId));
	toFile << "Particle Count      : " << aParticleCount     << std::endl;
	toFile << "Neighbor Pair Count : " << aNeighborPairCount << std::endl;
	toFile << "Simulation Step     : " << measurements::get_timing_interval_in_ms("Simulation Step"  ) << "ms" << std::endl;
	toFile << "Neighborhood Search : " << measurements::get_timing_interval_in_ms("Neighborhood"     ) << "ms" << std::endl;
	toFile << "Constraint Solver   : " << measurements::get_timing_interval_in_ms("Constraint Solver") << "ms" << std::endl;
	toFile << "Neighbor Search Type: " << NEIGHBOR_SEARCH_FILENAME << std::endl;
}
