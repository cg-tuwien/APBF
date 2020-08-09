#include "test.h"
#include "algorithms.h"
#include "neighborhood_brute_force.h"
#include "../shaders/cpu_gpu_shared_config.h"

void pbd::test::test_all()
{
	auto failCount = 0u;
	if (!gpu_list_concatenation_1())               failCount++;
	if (!gpu_list_concatenation_2())               failCount++;
	if (!gpu_list_apply_edit())                    failCount++;
	if (!indexed_list_write_increasing_sequence()) failCount++;
	if (!indexed_list_apply_hidden_edit_1())       failCount++;
	if (!indexed_list_apply_hidden_edit_2())       failCount++;
	if (!indexed_list_apply_hidden_edit_3())       failCount++;
	if (!prefix_sum())                             failCount++;
	if (!long_prefix_sum())                        failCount++;
	if (!very_long_prefix_sum())                   failCount++;
	if (!sort())                                   failCount++;
	if (!sort_many_values())                       failCount++;
	if (!sort_small_values())                      failCount++;
	if (!sort_many_small_values())                 failCount++;
	if (!delete_these_1())                         failCount++;
	if (!delete_these_2())                         failCount++;
	if (!neighborhood_brute_force())               failCount++;
//	if (!sortByPositions())                        failCount++;
//	if (!merge())                                  failCount++;
//	if (!mergeGenerator())                         failCount++;
//	if (!mergeGeneratorGrid())                     failCount++;

	if (failCount > 0u)
	{
		LOG_ERROR(std::to_string(failCount) + " TEST" + (failCount == 1u ? "" : "S") + " FAILED");
	}
}

bool pbd::test::gpu_list_concatenation_1()
{
	shader_provider::start_recording();
	auto listAData = std::vector<uint32_t>({3u, 64u, 12683u, 4294967295u});
	auto listBData = std::vector<uint32_t>({432587u, 0u, 5436u});
	auto listA = to_gpu_list(listAData).request_length(listAData.size() + listBData.size());
	auto listB = to_gpu_list(listBData);
	listA += listB;
	listAData.insert(listAData.end(), listBData.begin(), listBData.end());
	shader_provider::end_recording();
	return validate_list(listA.buffer(), listAData, "gpu_list concatenation");
}

bool pbd::test::gpu_list_concatenation_2()
{
	shader_provider::start_recording();
	auto listAData = std::vector<glm::vec3>({ glm::vec3(0, 2, 61.5), glm::vec3(13.65, 4.65, 234) });
	auto listBData = std::vector<glm::vec3>({ glm::vec3(1, 0, 2.5) });
	auto listCData = std::vector<glm::vec3>({ glm::vec3(8, 2, 1), glm::vec3(2, 4, 9) });
	auto listA = to_gpu_list(listAData);
	auto listB = to_gpu_list(listBData).request_length(listAData.size() + listBData.size() + listCData.size());
	auto listC = to_gpu_list(listCData);
	listB = listA + listB + listC;
	listAData.insert(listAData.end(), listBData.begin(), listBData.end());
	listAData.insert(listAData.end(), listCData.begin(), listCData.end());
	shader_provider::end_recording();
	return validate_list(listB.buffer(), listAData, "gpu_list concatenation 2");
}

bool pbd::test::gpu_list_apply_edit()
{
	shader_provider::start_recording();
	auto listData       = std::vector<uint32_t>({ 77u, 3u, 9999u, 4294967295u, 0u });
	auto editListData   = std::vector<uint32_t>({ 1u, 3u, 1u, 4u });
	auto expectedResult = std::vector<uint32_t>({ 3u, 4294967295u, 3u, 0u });
	auto     list = to_gpu_list(listData);
	auto editList = to_gpu_list(editListData);
	list.apply_edit(editList, nullptr);
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_length(list.length(), editListData.size(), "gpu_list::apply_edit() length") && pass;
	pass = validate_list(list.buffer(), expectedResult, "gpu_list::apply_edit()") && pass;
	return pass;
}

bool pbd::test::indexed_list_write_increasing_sequence()
{
	shader_provider::start_recording();
	auto expectedResult = std::vector<uint32_t>({ 0u, 1u, 2u });
	auto list = indexed_list<gpu_list<4ui64>>(5).request_length(3);
	list.increase_length(3);
	shader_provider::end_recording();
	return validate_list(list.index_buffer(), expectedResult, "IndexedList::increaseLength()");
}

bool pbd::test::indexed_list_apply_hidden_edit_1()
{
	shader_provider::start_recording();
	auto expectedResultA = std::vector<uint32_t>({ 0u, 1u, 2u, 3u });
	auto expectedResultB = std::vector<uint32_t>({0u });
	auto editListData    = std::vector<uint32_t>({ 1u, 2u, 3u, 4u });
	auto listA    = indexed_list<gpu_list<4ui64>>(5).request_length(5);
	auto editList = to_gpu_list(editListData);
	listA.increase_length(2);
	auto listB = listA; // 3, 2
	listA.increase_length(3);
	listA.hidden_list().apply_edit(editList, nullptr);
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_length(listA.length(), expectedResultA.size(), "IndexedList::applyHiddenEdit() 1 Length A") && pass;
	pass = validate_length(listB.length(), expectedResultB.size(), "IndexedList::applyHiddenEdit() 1 Length B") && pass;
	pass = validate_list(listA.index_buffer(), expectedResultA, "IndexedList::applyHiddenEdit() 1 A") && pass;
	pass = validate_list(listB.index_buffer(), expectedResultB, "IndexedList::applyHiddenEdit() 1 B") && pass;
	return pass;
}

bool pbd::test::indexed_list_apply_hidden_edit_2()
{
	shader_provider::start_recording();
	auto expectedResultA = std::vector<uint32_t>({ 0u, 1u, 2u, 3u });
	auto expectedResultB = std::vector<uint32_t>({ 0u, 1u, 2u });
	auto editListData    = std::vector<uint32_t>({ 0u, 1u, 3u, 4u });
	auto listBIdxData    = std::vector<uint32_t>({ 0u, 3u, 1u });
	auto listA    = indexed_list<gpu_list<12ui64>>(5).request_length(5);
	auto listB    = indexed_list<gpu_list<12ui64>>().set_length(listBIdxData.size());
	auto editList = to_gpu_list(editListData);
	listB.share_hidden_data_from(listA);
	listA.increase_length(5);
	algorithms::copy_bytes(listBIdxData.data(), listB.write().index_buffer(), listBIdxData.size() * 4);
	listA.hidden_list().apply_edit(editList, nullptr);
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_length(listA.hidden_list().length(), 4, "IndexedList::applyHiddenEdit() 2 Length Hidden List") && pass;
	pass = validate_length(listA.length(), expectedResultA.size(), "IndexedList::applyHiddenEdit() 2 Length A") && pass;
	pass = validate_length(listB.length(), expectedResultB.size(), "IndexedList::applyHiddenEdit() 2 Length B") && pass;
	pass = validate_list(listA.index_buffer(), expectedResultA, "IndexedList::applyHiddenEdit() 2 A") && pass;
	pass = validate_list(listB.index_buffer(), expectedResultB, "IndexedList::applyHiddenEdit() 2 B") && pass;
	return pass;
}

bool pbd::test::indexed_list_apply_hidden_edit_3()
{
	shader_provider::start_recording();
	auto expectedResultA = std::vector<uint32_t>({ 0u, 1u, 2u, 3u, 4u });
	auto expectedResultB = std::vector<uint32_t>({ 0u, 2u, 3u, 3u });
	auto editListData    = std::vector<uint32_t>({ 2u, 1u, 2u, 4u, 1u });
	auto listBIdxData    = std::vector<uint32_t>({ 4u, 2u, 4u });
	auto listA    = indexed_list<gpu_list<12ui64>>(5).request_length(5);
	auto listB    = indexed_list<gpu_list<12ui64>>().set_length(listBIdxData.size());
	auto editList = to_gpu_list(editListData);
	listB.share_hidden_data_from(listA);
	listA.increase_length(5);
	algorithms::copy_bytes(listBIdxData.data(), listB.write().index_buffer(), listBIdxData.size() * 4);
	listA.hidden_list().apply_edit(editList, nullptr);
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_length(listA.hidden_list().length(), 5, "IndexedList::applyHiddenEdit() 3 Length Hidden List") && pass;
	pass = validate_length(listA.length(), expectedResultA.size(), "IndexedList::applyHiddenEdit() 3 Length A") && pass;
	pass = validate_length(listB.length(), expectedResultB.size(), "IndexedList::applyHiddenEdit() 3 Length B") && pass;
	pass = validate_list(listA.index_buffer(), expectedResultA, "IndexedList::applyHiddenEdit() 3 A") && pass;
	pass = validate_list(listB.index_buffer(), expectedResultB, "IndexedList::applyHiddenEdit() 3 B") && pass;
	return pass;
}

bool pbd::test::prefix_sum()
{
	shader_provider::start_recording();
	auto listData       = std::vector<uint32_t>({ 43u, 1u, 4567u, 0u, 1u, 0u, 84523487u });
	auto expectedResult = std::vector<uint32_t>({ 43u, 44u, 4611u, 4611u, 4612u, 4612u, 84528099u });
	auto list           = to_gpu_list(listData);
	auto prefixHelper   = pbd::gpu_list<4ui64>();
	prefixHelper.request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(listData.size()));
	algorithms::prefix_sum(list.write().buffer(), prefixHelper.write().buffer(), list.write().length());
	shader_provider::end_recording();
	return validate_list(list.buffer(), expectedResult, "prefix sum");
}

bool pbd::test::long_prefix_sum()
{
	shader_provider::start_recording();
	srand(0);
	auto listData       = std::vector<uint32_t>();
	auto expectedResult = std::vector<uint32_t>();
	listData      .reserve(1000);
	expectedResult.reserve(1000);
	auto sum = 0u;

	for (auto i = 0u; i < 1000u; i++)
	{
		auto value = static_cast<uint32_t>(rand()) & 3u; // only use last two bits
		sum += value;
		listData      .push_back(value);
		expectedResult.push_back(sum);
	}

	auto list         = to_gpu_list(listData);
	auto prefixHelper = pbd::gpu_list<4ui64>();
	prefixHelper.request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(listData.size()));
	algorithms::prefix_sum(list.write().buffer(), prefixHelper.write().buffer(), list.write().length());
	shader_provider::end_recording();
	return validate_list(list.buffer(), expectedResult, "long prefix sum");
}

bool pbd::test::very_long_prefix_sum()
{
	shader_provider::start_recording();
	srand(0);
	auto blocksize = 512u;
	auto listLength = blocksize * blocksize + 1000u;
	auto helperLength = algorithms::prefix_sum_calculate_needed_helper_list_length(listLength);
	auto listData = std::vector<uint32_t>();
	auto expectedResult = std::vector<uint32_t>();
	auto expectedHelper = std::vector<uint32_t>();
	listData.reserve(listLength);
	expectedResult.reserve(listLength);
	expectedHelper.reserve(helperLength);
	auto sum = 0u;

	for (auto i = 0u; i < listLength; i++)
	{
		auto value = static_cast<uint32_t>(rand()) & 1u; // only use last bit
		sum += value;
		listData.push_back(value);
		expectedResult.push_back(sum);
	}
	for (auto i = 1u; i <= listLength / blocksize; i++)
	{
		expectedHelper.push_back(expectedResult[i * blocksize - 1u]);
	}
	expectedHelper.push_back(expectedResult.back());
	for (auto i = 1u; i <= (listLength + blocksize - 1u) / blocksize / blocksize; i++)
	{
		expectedHelper.push_back(expectedHelper[i * blocksize - 1u]);
	}
	expectedHelper.push_back(expectedResult.back());

	auto list         = to_gpu_list(listData);
	auto prefixHelper = pbd::gpu_list<4ui64>();
	prefixHelper.request_length(helperLength);
	algorithms::prefix_sum(list.write().buffer(), prefixHelper.write().buffer(), list.write().length());
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_list(        list.buffer(), expectedResult, "very long prefix sum"       ) && pass;
	pass = validate_list(prefixHelper.buffer(), expectedHelper, "very long prefix sum helper") && pass;
	return pass;
}

bool pbd::test::sort()
{
	shader_provider::start_recording();
	auto listData = std::vector<uint32_t>({ 15u, 2u, 1234u, 2u, 0u, 4294967295u, 1u, 4294967294u });
	auto expectedResult = std::vector<uint32_t>({ 0u, 1u, 2u, 2u, 15u, 1234u, 4294967294u, 4294967295u });
	auto indexData = std::vector<uint32_t>({ 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u });
	auto expectedIndexResult = std::vector<uint32_t>({ 4u, 6u, 1u, 3u, 0u, 2u, 7u, 5u });
	auto list        = to_gpu_list(listData);
	auto indices     = to_gpu_list(indexData);
	auto result      = pbd::gpu_list<4ui64>();
	auto indexResult = pbd::gpu_list<4ui64>();
	auto sortHelper  = pbd::gpu_list<4ui64>();
	result     .request_length(listData.size());
	indexResult.request_length(indexData.size());
	sortHelper .request_length(algorithms::sort_calculate_needed_helper_list_length(listData.size()));
	algorithms::sort(list.write().buffer(), indices.write().buffer(), sortHelper.write().buffer(), list.write().length(), result.write().buffer(), indexResult.write().buffer());
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_list(result.buffer(), expectedResult, "sort") && pass;
	pass = validate_list(indexResult.buffer(), expectedIndexResult, "sort indices") && pass;
	return pass;
}

bool pbd::test::sort_many_values()
{
	shader_provider::start_recording();
	srand(0);
	auto blocksize = 512u;
	auto listLength = blocksize * blocksize + 123u;
	auto listData = std::vector<uint32_t>();
	auto indexData = std::vector<uint32_t>();
	listData.reserve(listLength);
	indexData.reserve(listLength);

	for (auto i = 0u; i < listLength; i++)
	{
		auto value = static_cast<uint32_t>(rand());
		listData.push_back(value);
		indexData.push_back(i);
	}

	auto list        = to_gpu_list(listData);
	auto indices     = to_gpu_list(indexData);
	auto result      = pbd::gpu_list<4ui64>();
	auto indexResult = pbd::gpu_list<4ui64>();
	auto sortHelper  = pbd::gpu_list<4ui64>();
	result     .request_length(listData.size());
	indexResult.request_length(indexData.size());
	sortHelper .request_length(algorithms::sort_calculate_needed_helper_list_length(listData.size()));
	algorithms::sort(list.write().buffer(), indices.write().buffer(), sortHelper.write().buffer(), list.write().length(), result.write().buffer(), indexResult.write().buffer());
	shader_provider::end_recording();
	std::stable_sort(indexData.begin(), indexData.end(), [&listData](size_t i0, size_t i1) { return listData[i0] < listData[i1]; });
	std::sort(listData.begin(), listData.end());
	auto pass = true;
	pass = validate_list(result.buffer(), listData, "sort (many values)") && pass;
	pass = validate_list(indexResult.buffer(), indexData, "sort indices (many values)") && pass;
	return pass;
}

bool pbd::test::sort_small_values()
{
	shader_provider::start_recording();
	auto listData = std::vector<uint32_t>({ 15u, 2u, 3u, 2u, 0u, 14u, 1u, 14u });
	auto expectedResult = std::vector<uint32_t>({ 0u, 1u, 2u, 2u, 3u, 14u, 14u, 15u });
	auto indexData = std::vector<uint32_t>({ 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u });
	auto expectedIndexResult = std::vector<uint32_t>({ 4u, 6u, 1u, 3u, 2u, 5u, 7u, 0u });
	auto list        = to_gpu_list(listData);
	auto indices     = to_gpu_list(indexData);
	auto result      = pbd::gpu_list<4ui64>();
	auto indexResult = pbd::gpu_list<4ui64>();
	auto sortHelper  = pbd::gpu_list<4ui64>();
	result     .request_length(listData.size());
	indexResult.request_length(indexData.size());
	sortHelper .request_length(algorithms::sort_calculate_needed_helper_list_length(listData.size()));
	algorithms::sort(list.write().buffer(), indices.write().buffer(), sortHelper.write().buffer(), list.write().length(), result.write().buffer(), indexResult.write().buffer());
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_list(result.buffer(), expectedResult, "sort (small values)") && pass;
	pass = validate_list(indexResult.buffer(), expectedIndexResult, "sort indices (small values)") && pass;
	return pass;
}

bool pbd::test::sort_many_small_values()
{
	shader_provider::start_recording();
	srand(0);
	auto blocksize = 512u;
	auto listLength = blocksize * blocksize + 1000u;
	auto listData = std::vector<uint32_t>();
	auto indexData = std::vector<uint32_t>();
	listData.reserve(listLength);
	indexData.reserve(listLength);

	for (auto i = 0u; i < listLength; i++)
	{
		auto value = static_cast<uint32_t>(rand()) & 15u; // only use last 4 bits
		listData.push_back(value);
		indexData.push_back(i);
	}

	auto list        = to_gpu_list(listData);
	auto indices     = to_gpu_list(indexData);
	auto result      = pbd::gpu_list<4ui64>();
	auto indexResult = pbd::gpu_list<4ui64>();
	auto sortHelper  = pbd::gpu_list<4ui64>();
	result     .request_length(listData.size());
	indexResult.request_length(indexData.size());
	sortHelper .request_length(algorithms::sort_calculate_needed_helper_list_length(listData.size()));
	algorithms::sort(list.write().buffer(), indices.write().buffer(), sortHelper.write().buffer(), list.write().length(), result.write().buffer(), indexResult.write().buffer());
	shader_provider::end_recording();
	std::stable_sort(indexData.begin(), indexData.end(), [&listData](size_t i0, size_t i1) { return listData[i0] < listData[i1]; });
	std::sort(listData.begin(), listData.end());
	auto pass = true;
	pass = validate_list(result.buffer(), listData, "sort (many small values)") && pass;
	pass = validate_list(indexResult.buffer(), indexData, "sort indices (many small values)") && pass;
	return pass;
}

bool pbd::test::delete_these_1()
{
	shader_provider::start_recording();
	auto hiddenValues   = std::vector<uint32_t>({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 });
	auto expectedResult = std::vector<uint32_t>({ 12, 13 });
	auto listA = indexed_list<gpu_list<4ui64>>(13).request_length(13);
	listA.increase_length(8);
	auto listB = listA;
	auto listC = listA;
	listB.increase_length(3);
	listC.increase_length(2);
	algorithms::copy_bytes(hiddenValues.data(), listA.hidden_list().write().buffer(), hiddenValues.size() * 4);
	listB.delete_these();
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_length(listA.length(), 0, "delete_these() length A") && pass;
	pass = validate_length(listB.length(), 0, "delete_these() length B") && pass;
	pass = validate_length(listC.length(), 2, "delete_these() length C") && pass;
	pass = validate_length(listA.hidden_list().length(), 2, "delete_these() length Hidden List") && pass;
	pass = validate_list(listC.hidden_list().buffer(), expectedResult, "delete_these()") && pass;
	return pass;
}

bool pbd::test::delete_these_2()
{
	shader_provider::start_recording();
	auto listA = indexed_list<gpu_list<4ui64>>(13).request_length(13);
	listA.increase_length(8);
	auto listB = listA;
	listB.increase_length(5);
	listB.delete_these();
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_length(listA.length(), 0, "delete_these() 2 length A") && pass;
	pass = validate_length(listB.length(), 0, "delete_these() 2 length B") && pass;
	pass = validate_length(listA.hidden_list().length(), 0, "delete_these() 2 length Hidden List") && pass;
	return pass;
}

bool pbd::test::neighborhood_brute_force()
{
	shader_provider::start_recording();
	auto positionsData = std::vector<glm::ivec4>({ glm::ivec4(0, 0, 0, 1), glm::ivec4(262144, 0, 0, 1), glm::ivec4(0, 262144, 0, 1) });
	auto rangeData     = std::vector<float>({ 1.0f, 1.0f, 2.0f });
	auto expectedNeighbors1 = std::vector<uint32_t>({ 3, 0, 1, 2 });
	auto expectedNeighbors2 = std::vector<uint32_t>({ 2, 0, 1 });
	auto expectedNeighbors3 = std::vector<uint32_t>({ 3, 0, 1, 2 });
	auto range = to_gpu_list(rangeData);
	auto particles = pbd::particles(positionsData.size());
	auto neighbors = pbd::gpu_list<sizeof(uint32_t) * 64>().request_length(positionsData.size() * 64);
	auto neighborhoodBruteForce = pbd::neighborhood_brute_force();
	particles.request_length(positionsData.size());
	particles.increase_length(positionsData.size());
	particles.hidden_list().get<hidden_particles::id::position>() = to_gpu_list(positionsData);
	neighborhoodBruteForce.set_data(&particles, &range, &neighbors).set_range_scale(1).apply();
	shader_provider::end_recording();
	auto pass = true;
	pass = validate_list(neighbors.buffer(), expectedNeighbors1, "neighborhood_brute_force", 0 * NEIGHBOR_LIST_MAX_LENGTH) && pass;
	pass = validate_list(neighbors.buffer(), expectedNeighbors2, "neighborhood_brute_force", 1 * NEIGHBOR_LIST_MAX_LENGTH) && pass;
	pass = validate_list(neighbors.buffer(), expectedNeighbors3, "neighborhood_brute_force", 2 * NEIGHBOR_LIST_MAX_LENGTH) && pass;
	return pass;
}

/*bool pbd::test::sortByPositions() // TODO fix test case
{
	auto positions = std::vector<glm::int3>({ glm::int3(32768, 81920, 16384), glm::int3(123456, 0, 57), glm::int3(16383, 16384, 57), glm::int3(-1, 1, -16385) }); // 2:5:1,7:0:0,0:1:0,63:0:62 => 142,73,2,187241
	auto phases = std::vector<uint32_t>({ 0u, 1u, 2u, 3u });
	auto sortedPhases = std::vector<uint32_t>({ 2u, 1u, 0u, 3u });
	auto cellStart = std::vector<uint32_t>();
	auto cellEnd   = std::vector<uint32_t>();
	auto particles = Particles(9);
	cellStart.assign(static_cast<size_t>(pow(64u, 3u)), 0u);
	cellEnd.assign(static_cast<size_t>(pow(64u, 3u)), 0u);
	cellStart[   142] = 2u;               cellEnd[   142] = 3u;
	cellStart[    73] = 1u;               cellEnd[    73] = 2u;
	cellStart[     2] = 0u;               cellEnd[     2] = 1u;
	cellStart[187241] = 3u;               cellEnd[187241] = 4u;
	particles.increaseLength(positions.size());
	fillList(particles.getHiddenList().get<HiddenParticles::id::position>().getWriteBuffer(), positions);
	fillList(particles.getHiddenList().get<HiddenParticles::id::phase>().getWriteBuffer(), phases);
	auto sortByPos = std::make_unique<SortByPositions>();
	sortByPos->setParticles(particles);
	sortByPos->apply(0.0f);
	auto neighborhoodInfo = sortByPos->getNeighborhoodInfo();
	auto pass = validateList(particles.getHiddenList().get<HiddenParticles::id::phase>().getReadBuffer(), sortedPhases, "sortByPositions phases");
	pass = validateList(neighborhoodInfo->get<NeighborhoodInfo::id::cellStart>().getReadBuffer(), cellStart, "sortByPositions cellStart") && pass;
	pass = validateList(neighborhoodInfo->get<NeighborhoodInfo::id::cellEnd  >().getReadBuffer(), cellEnd  , "sortByPositions cellEnd"  ) && pass;
	return pass;
}

bool pbd::test::merge()
{
	auto positions          = std::vector<glm::int3>({ glm::int3(1, -2, 3), glm::int3(-4, 5, -6), glm::int3(7, -8, 9), glm::int3(-10, 11, -12), glm::int3(13, -14, 15) });
	auto velocities         = std::vector<glm::float3>({ glm::float3(1.5f, -2.5f, 3.5f), glm::float3(-4.5f, 5.5f, -6.5f), glm::float3(7.5f, -8.5f, 9.5f), glm::float3(-10.5f, 11.5f, -12.5f), glm::float3(13.5f, -14.5f, 15.5f) });
	auto masses             = std::vector<float>({ 1.5f, 3.0f, 6.25f, 0.75f, 1.0f });
	auto radii              = std::vector<float>({ 1.0f, 3.0f, 1.25f, 1.75f, 2.0f });
	auto listA              = std::vector<uint32_t>({ 2, 4, 2 });
	auto listB              = std::vector<uint32_t>({ 1, 3, 0 });
	auto expectedListA      = std::vector<uint32_t>({ 3 });
	auto expectedListB      = std::vector<uint32_t>({ 2 });
	auto expectedPositions  = std::vector<glm::int3>({ glm::int3(1, -2, 3), glm::int3(3, -3, 4), glm::int3(-10, 11, -12), glm::int3(13, -14, 15) });
	auto expectedMasses     = std::vector<float>({ 1.5f, 9.25f, 0.75f, 1.0f });
	auto expectedVelocities = std::vector<glm::float3>({ glm::float3(1.5f, -2.5f, 3.5f), (masses[1] * glm::float3(-4.5f, 5.5f, -6.5f) + masses[2] * glm::float3(7.5f, -8.5f, 9.5f)) / (masses[1] + masses[2]), glm::float3(-10.5f, 11.5f, -12.5f), glm::float3(13.5f, -14.5f, 15.5f) });
	auto expectedRadii      = std::vector<float>({ 1.0f, 3.07066059f, 1.75f, 2.0f });

	auto allParticles = Particles(5);
	allParticles.increaseLength(5);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::position>().getWriteBuffer(), positions);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::posBackup>().getWriteBuffer(), positions);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::velocity>().getWriteBuffer(), velocities);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::mass>().getWriteBuffer(), masses);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::radius>().getWriteBuffer(), radii);

	auto mergeManipulator = std::make_unique<Merge>();
	auto& mergeList = mergeManipulator->getList();
	mergeList.get<MergeList::id::particleA>().shareHiddenDataFrom(allParticles);
	mergeList.get<MergeList::id::particleB>().shareHiddenDataFrom(allParticles);
	mergeList.resize(3);
	fillList(mergeList.get<MergeList::id::particleA>().getIndexWriteBuffer(), listA);
	fillList(mergeList.get<MergeList::id::particleB>().getIndexWriteBuffer(), listB);

	mergeManipulator->apply(0.1f);

	auto pass = mergeList.length() == 1 && allParticles.length() == 4 && allParticles.getHiddenList().length() == 4;
	pass = validateList(mergeList.get<MergeList::id::particleA>().getIndexReadBuffer(), expectedListA, "merge mergeList A") && pass;
	pass = validateList(mergeList.get<MergeList::id::particleB>().getIndexReadBuffer(), expectedListB, "merge mergeList B") && pass;
	pass = validateList(allParticles.getHiddenList().get<HiddenParticles::id::position>().getReadBuffer(), expectedPositions, "merge positions") && pass;
	pass = validateList(allParticles.getHiddenList().get<HiddenParticles::id::velocity>().getReadBuffer(), expectedVelocities, "merge velocities") && pass;
	pass = validateList(allParticles.getHiddenList().get<HiddenParticles::id::mass>().getReadBuffer(), expectedMasses, "merge masses") && pass;
	pass = validateList(allParticles.getHiddenList().get<HiddenParticles::id::radius>().getReadBuffer(), expectedRadii, "merge radii") && pass;
	return pass;
}

bool pbd::test::mergeGenerator()
{
	auto positions          = std::vector<glm::int3>({ glm::int3(0, 0, 0), glm::int3(2, 0, 0), glm::int3(7, 2, 0), glm::int3(0, 2, 0), glm::int3(5, 0, 0) });
	auto velocities         = std::vector<glm::float3>({ glm::float3(1.0f, 0.0f, 0.0f), glm::float3(0.9f, 0.0f, 0.0f), glm::float3(0.7f, 0.0f, 0.0f), glm::float3(1.0f, 0.0f, 0.0f), glm::float3(0.8f, 0.0f, 0.0f) });
	auto masses             = std::vector<float>({ 1.0f, 1.0f, 1.0f, 1.0f, 8.0f });
	auto radii              = std::vector<float>({ 1.0f, 1.0f, 1.0f, 1.0f, 2.0f });
	auto expectedListA      = std::vector<uint32_t>({ 3, 1, 4, 2, 0 });
	auto expectedListB      = std::vector<uint32_t>({ 0, 0, 0, 2, 0 });

	for (auto& pos : positions) pos *= 262144.0f;

	auto allParticles = Particles(5);
	allParticles.increaseLength(5);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::position>().getWriteBuffer(), positions);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::posBackup>().getWriteBuffer(), positions);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::velocity>().getWriteBuffer(), velocities);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::mass>().getWriteBuffer(), masses);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::radius>().getWriteBuffer(), radii);

	auto pMergeManipulator = std::make_shared<pbd::Merge>();
	auto pMergeGenerator = std::make_shared<pbd::GMerge>(pMergeManipulator);
	pMergeGenerator->addParticles(allParticles);

	auto& mergeList = pMergeManipulator->getList();

	static_cast<std::shared_ptr<ParticleManipulatorGenerator>>(pMergeGenerator)->generate();

	auto pass = mergeList.length() == 5 && allParticles.length() == 5 && allParticles.getHiddenList().length() == 5;
	pass = validateList(mergeList.get<MergeList::id::particleA>().getIndexReadBuffer(), expectedListA, "merge generator mergeList A") && pass;
	pass = validateList(mergeList.get<MergeList::id::particleB>().getIndexReadBuffer(), expectedListB, "merge generator mergeList B") && pass;
	return pass;
}

bool pbd::test::mergeGeneratorGrid()
{
	auto gridSize = glm::int3(5, 5, 1);
	auto particleCount = gridSize.x * gridSize.y * gridSize.z;
	auto positions = std::vector<glm::int3>();
	auto velocities = std::vector<glm::float3>();
	auto masses = std::vector<float>();
	auto radii = std::vector<float>();
	auto expectedListA = std::vector<uint32_t>({ 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 18, 16, 8, 6, 24, 22, 20, 14, 12, 10, 4, 2, 0 });
	auto expectedListB = std::vector<uint32_t>({ 22, 20, 14, 12, 10, 12, 10, 4, 2, 0, 2, 0, 12, 10, 2, 0, 24, 22, 20, 14, 12, 10, 4, 2, 0 });

	for (auto x = 0; x < gridSize.x; x++) for (auto y = 0; y < gridSize.y; y++) for (auto z = 0; z < gridSize.z; z++)
	{
		positions.push_back(glm::int3(x, y, z) * 2);
		velocities.push_back(glm::float3(0.0f));
		masses.push_back(1.0f);
		radii.push_back(1.0f);
	}
	for (auto& pos : positions) pos *= 262144.0f;

	auto allParticles = Particles(particleCount);
	allParticles.increaseLength(particleCount);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::position>().getWriteBuffer(), positions);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::posBackup>().getWriteBuffer(), positions);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::velocity>().getWriteBuffer(), velocities);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::mass>().getWriteBuffer(), masses);
	fillList(allParticles.getHiddenList().get<HiddenParticles::id::radius>().getWriteBuffer(), radii);

	auto pMergeManipulator = std::make_shared<pbd::Merge>();
	auto pMergeGenerator = std::make_shared<pbd::GMerge>(pMergeManipulator);
	pMergeGenerator->addParticles(allParticles);

	auto& mergeList = pMergeManipulator->getList();

	static_cast<std::shared_ptr<ParticleManipulatorGenerator>>(pMergeGenerator)->generate();

	auto pass = mergeList.length() == particleCount && allParticles.length() == particleCount && allParticles.getHiddenList().length() == particleCount;
	pass = validateList(mergeList.get<MergeList::id::particleA>().getIndexReadBuffer(), expectedListA, "merge generator grid mergeList A") && pass;
	pass = validateList(mergeList.get<MergeList::id::particleB>().getIndexReadBuffer(), expectedListB, "merge generator grid mergeList B") && pass;
	return pass;
}*/

bool pbd::test::validate_length(const avk::buffer& aLength, size_t aExpectedLength, const std::string& aTestName)
{
	auto data = 0u;
	aLength->read(&data, 0, avk::sync::wait_idle(true));
	if (data != aExpectedLength) {
		LOG_WARNING("TEST FAIL: [" + aTestName + "] - expected length " + std::to_string(aExpectedLength) + " but got " + std::to_string(data));
		return false;
	}
	return true;
}
