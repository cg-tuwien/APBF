#include "test.h"
#include "gpu_list.h"

void pbd::test::test_all()
{
	auto failCount = 0u;
	if (!gpu_list_concatenation())              failCount++;
	if (!gpu_list_concatenation2())             failCount++;
	if (!gpu_list_apply_edit())                 failCount++;
/*	if (!indexedList_writeDecreasingSequence()) failCount++;
	if (!indexedList_applyHiddenEdit())         failCount++;
	if (!indexedList_applyHiddenEdit2())        failCount++;
	if (!indexedList_applyHiddenEdit3())        failCount++;
	if (!prefixSum())                           failCount++;
	if (!longPrefixSum())                       failCount++;
	if (!veryLongPrefixSum())                   failCount++;
	if (!sort())                                failCount++;
	if (!sortManyValues())                      failCount++;
	if (!sortSmallValues())                     failCount++;
	if (!sortManySmallValues())                 failCount++;
//	if (!sortByPositions())                     failCount++;
	if (!deleteThese())                         failCount++;
//	if (!merge())                               failCount++;
//	if (!mergeGenerator())                      failCount++;
//	if (!mergeGeneratorGrid())                  failCount++;*/

	if (failCount > 0u)
	{
		LOG_ERROR(std::to_string(failCount) + " TEST" + (failCount == 1u ? "" : "S") + " FAILED");
	}
}

void pbd::test::test_quick()
{
	auto failCount = 0u;
	if (!gpu_list_concatenation())              failCount++;
	if (!gpu_list_concatenation2())             failCount++;
	if (!gpu_list_apply_edit())                 failCount++;
/*	if (!indexedList_writeDecreasingSequence()) failCount++;
	if (!indexedList_applyHiddenEdit())         failCount++;
	if (!indexedList_applyHiddenEdit2())        failCount++;
	if (!indexedList_applyHiddenEdit3())        failCount++;
	if (!prefixSum())                           failCount++;
	if (!sort())                                failCount++;
//	if (!sortByPositions())                     failCount++;
	if (!deleteThese())                         failCount++;
//	if (!merge())                               failCount++;
//	if (!mergeGenerator())                      failCount++;
//	if (!mergeGeneratorGrid())                  failCount++;*/

	if (failCount > 0u)
	{
		LOG_ERROR(std::to_string(failCount) + " TEST" + (failCount == 1u ? "" : "S") + " FAILED");
	}
}

bool pbd::test::gpu_list_concatenation()
{
	shader_provider::start_recording();
	auto listAData = std::vector<uint32_t>({3u, 64u, 12683u, 4294967295u});
	auto listBData = std::vector<uint32_t>({432587u, 0u, 5436u});
	auto listA = pbd::gpu_list<4>();
	auto listB = pbd::gpu_list<4>();
	listA.set_length(listAData.size());
	listB.set_length(listBData.size());
	listA.request_length(listAData.size() + listBData.size());
	listA.write_buffer()->fill(listAData.data(), 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr()));
	listB.write_buffer()->fill(listBData.data(), 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr()));
	listA += listB;
	listAData.insert(listAData.end(), listBData.begin(), listBData.end());
	shader_provider::end_recording();
	return validate_list(listA.read_buffer(), listAData, "gpu_list concatenation");
}

bool pbd::test::gpu_list_concatenation2()
{
	shader_provider::start_recording();
	auto listAData = std::vector<glm::vec3>({ glm::vec3(0, 2, 61.5), glm::vec3(13.65, 4.65, 234) });
	auto listBData = std::vector<glm::vec3>({ glm::vec3(1, 0, 2.5) });
	auto listCData = std::vector<glm::vec3>({ glm::vec3(8, 2, 1), glm::vec3(2, 4, 9) });
	auto listA = pbd::gpu_list<12>();
	auto listB = pbd::gpu_list<12>();
	auto listC = pbd::gpu_list<12>();
	listA.set_length(listAData.size());
	listB.set_length(listBData.size());
	listC.set_length(listCData.size());
	listB.request_length(listAData.size() + listBData.size() + listCData.size());
	listA.write_buffer()->fill(listAData.data(), 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr()));
	listB.write_buffer()->fill(listBData.data(), 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr()));
	listC.write_buffer()->fill(listCData.data(), 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr()));
	listB = listA + listB + listC;
	listAData.insert(listAData.end(), listBData.begin(), listBData.end());
	listAData.insert(listAData.end(), listCData.begin(), listCData.end());
	shader_provider::end_recording();
	return validate_list(listB.read_buffer(), listAData, "gpu_list concatenation 2");
}

bool pbd::test::gpu_list_apply_edit()
{
	/*auto listData       = std::vector<uint32_t>({ 77u, 3u, 9999u, 4294967295u, 0u });
	auto editListData   = std::vector<uint32_t>({ 1u, 3u, 1u, 4u });
	auto expectedResult = std::vector<uint32_t>({ 3u, 4294967295u, 3u, 0u });
	auto list     = GpuList<GpuListType::Uint>();
	auto editList = GpuList<GpuListType::Uint>();
	    list.resize(    listData.size());
	editList.resize(editListData.size());
	fillList(    list.getWriteBuffer(),     listData);
	fillList(editList.getWriteBuffer(), editListData);
	list.applyEdit(editList, nullptr);
	auto result = list.length() == 4;
	if (!result)
	{
		logWarning("TEST FAIL: GpuList::applyEdit() - expected list length 4 but got list length " + std::to_string(list.length()));
	}
	return validateList(list.getReadBuffer(), expectedResult, "GpuList::applyEdit()") && result;*/
	return true;
}

/*bool pbd::test::indexedList_writeDecreasingSequence()
{
	auto expectedResult = std::vector<uint32_t>({ 2u, 1u, 0u });
	auto list = IndexedList<GpuList<GpuListType::Float>>(5);
	list.increaseLength(3);
	return validateList(list.getIndexReadBuffer(), expectedResult, "IndexedList::increaseLength()");
}

bool pbd::test::indexedList_applyHiddenEdit()
{
	auto expectedResultA = std::vector<uint32_t>({ 0u, 1u, 2u, 3u });
	auto expectedResultB = std::vector<uint32_t>({ 1u, 2u });
	auto listA = IndexedList<GpuList<GpuListType::Float>>(5);
	auto editList = GpuList<GpuListType::Uint>();
	editList.resize(4);
	fillList(editList.getWriteBuffer(), std::vector<uint32_t>({ 1u, 2u, 3u, 4u }));
	listA.increaseLength(5);
	auto listB = listA.subset(1, 2); // 3, 2
	listA.getHiddenList().applyEdit(editList, nullptr);
	auto pass = validateList(listA.getIndexReadBuffer(), expectedResultA, "IndexedList::applyHiddenEdit()");
	pass = validateList(listB.getIndexReadBuffer(), expectedResultB, "IndexedList::applyHiddenEdit()") && pass;
	return pass;
}

bool pbd::test::indexedList_applyHiddenEdit2()
{
	auto expectedResultB = std::vector<uint32_t>({ 0u, 1u, 2u });
	auto listA = IndexedList<GpuList<GpuListType::Float>>(5);
	auto editList = GpuList<GpuListType::Uint>();
	editList.resize(4);
	listA.increaseLength(5);
	auto listB = listA.subset(0, 3);
	fillList(editList.getWriteBuffer(), std::vector<uint32_t>({ 0u, 1u, 3u, 4u }));
	fillList(listB.getIndexWriteBuffer(), std::vector<uint32_t>({ 0u, 3u, 1u }));
	listA.getHiddenList().applyEdit(editList, nullptr);
	auto pass = listB.length() == 3 &&listA.length() == 4 && listA.getHiddenList().length() == 4;
	pass = validateList(listB.getIndexReadBuffer(), expectedResultB, "IndexedList::applyHiddenEdit() - second test") && pass;
	return pass;
}

bool pbd::test::indexedList_applyHiddenEdit3()
{
	auto expectedResultA = std::vector<uint32_t>({ 0u, 1u, 2u, 3u, 4u });
	auto expectedResultB = std::vector<uint32_t>({ 0u, 2u, 3u, 3u });
	auto listA = IndexedList<GpuList<GpuListType::Float>>(5);
	auto listB = IndexedList<GpuList<GpuListType::Float>>(0);
	auto editList = GpuList<GpuListType::Uint>();
	listB.shareHiddenDataFrom(listA);
	editList.resize(5);
	listA.increaseLength(5);
	fillList(editList.getWriteBuffer(), std::vector<uint32_t>({ 2u, 1u, 2u, 4u, 1u }));
	fillList(listB.getIndexWriteBuffer(3, false), std::vector<uint32_t>({ 4u, 2u, 4u }));
	listA.getHiddenList().applyEdit(editList, nullptr);
	auto pass = listA.length() == 5 && listB.length() == 4 && listB.getHiddenList().length() == 5;
	pass = validateList(listA.getIndexReadBuffer(), expectedResultA, "IndexedList::applyHiddenEdit() - third test") && pass;
	pass = validateList(listB.getIndexReadBuffer(), expectedResultB, "IndexedList::applyHiddenEdit() - third test") && pass;
	return pass;
}

bool pbd::test::prefixSum()
{
	auto listData       = std::vector<uint32_t>({ 43u, 1u, 4567u, 0u, 1u, 0u, 84523487u });
	auto expectedResult = std::vector<uint32_t>({ 43u, 44u, 4611u, 4611u, 4612u, 4612u, 84528099u });
	auto list         = GpuList<GpuListType::Uint>();
	auto prefixHelper = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	prefixHelper.resize(ListManipulation::prefixSum_calculateNeededHelperListLength(list.length()));
	fillList(list.getWriteBuffer(), listData);
	ListManipulation::prefixSum(list.getWriteBuffer(), prefixHelper.getWriteBuffer(), list.length());
	return validateList(list.getReadBuffer(), expectedResult, "prefix sum");
}

bool pbd::test::longPrefixSum()
{
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

	auto list         = GpuList<GpuListType::Uint>();
	auto prefixHelper = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	prefixHelper.resize(ListManipulation::prefixSum_calculateNeededHelperListLength(list.length()));
	fillList(list.getWriteBuffer(), listData);
	ListManipulation::prefixSum(list.getWriteBuffer(), prefixHelper.getWriteBuffer(), list.length());
	return validateList(list.getReadBuffer(), expectedResult, "long prefix sum");
}

bool pbd::test::veryLongPrefixSum()
{
	srand(0);
	auto blocksize = 512u;
	auto listLength = blocksize * blocksize + 1000u;
	auto helperLength = ListManipulation::prefixSum_calculateNeededHelperListLength(listLength);
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

	auto list         = GpuList<GpuListType::Uint>();
	auto prefixHelper = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	prefixHelper.resize(helperLength);
	fillList(list.getWriteBuffer(), listData);
	ListManipulation::prefixSum(list.getWriteBuffer(), prefixHelper.getWriteBuffer(), list.length());
	auto pass = validateList(list.getReadBuffer(), expectedResult, "very long prefix sum");
	pass = validateList(prefixHelper.getReadBuffer(), expectedHelper, "very long prefix sum helper") && pass;
	return pass;
}

bool pbd::test::sort()
{
	auto listData = std::vector<uint32_t>({ 15u, 2u, 1234u, 2u, 0u, 4294967295u, 1u, 4294967294u });
	auto expectedResult = std::vector<uint32_t>({ 0u, 1u, 2u, 2u, 15u, 1234u, 4294967294u, 4294967295u });
	auto indexData = std::vector<uint32_t>({ 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u });
	auto expectedIndexResult = std::vector<uint32_t>({ 4u, 6u, 1u, 3u, 0u, 2u, 7u, 5u });
	auto list        = GpuList<GpuListType::Uint>();
	auto result      = GpuList<GpuListType::Uint>();
	auto indices     = GpuList<GpuListType::Uint>();
	auto indexResult = GpuList<GpuListType::Uint>();
	auto sortHelper  = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	result.resize(listData.size());
	indices.resize(indexData.size());
	indexResult.resize(indexData.size());
	sortHelper.resize(ListManipulation::sort_calculateNeededHelperListLength(list.length()));
	fillList(list.getWriteBuffer(), listData);
	fillList(indices.getWriteBuffer(), indexData);
	ListManipulation::sort(list.getWriteBuffer(), indices.getWriteBuffer(), sortHelper.getWriteBuffer(), list.length(), result.getWriteBuffer(), indexResult.getWriteBuffer());
	auto pass = validateList(result.getReadBuffer(), expectedResult, "sort");
	return pass && validateList(indexResult.getReadBuffer(), expectedIndexResult, "sort indices");
}

bool pbd::test::sortManyValues()
{
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
	
	auto list        = GpuList<GpuListType::Uint>();
	auto result      = GpuList<GpuListType::Uint>();
	auto indices     = GpuList<GpuListType::Uint>();
	auto indexResult = GpuList<GpuListType::Uint>();
	auto sortHelper  = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	result.resize(listData.size());
	indices.resize(indexData.size());
	indexResult.resize(indexData.size());
	sortHelper.resize(ListManipulation::sort_calculateNeededHelperListLength(list.length()));
	fillList(list.getWriteBuffer(), listData);
	fillList(indices.getWriteBuffer(), indexData);
	ListManipulation::sort(list.getWriteBuffer(), indices.getWriteBuffer(), sortHelper.getWriteBuffer(), list.length(), result.getWriteBuffer(), indexResult.getWriteBuffer());
	std::stable_sort(indexData.begin(), indexData.end(), [&listData](size_t i0, size_t i1) { return listData[i0] < listData[i1]; });
	std::sort(listData.begin(), listData.end());
	auto pass = validateList(result.getReadBuffer(), listData, "sort (many values)");
	return pass && validateList(indexResult.getReadBuffer(), indexData, "sort indices (many values)");
}

bool pbd::test::sortSmallValues()
{
	auto listData = std::vector<uint32_t>({ 15u, 2u, 3u, 2u, 0u, 14u, 1u, 14u });
	auto expectedResult = std::vector<uint32_t>({ 0u, 1u, 2u, 2u, 3u, 14u, 14u, 15u });
	auto indexData = std::vector<uint32_t>({ 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u });
	auto expectedIndexResult = std::vector<uint32_t>({ 4u, 6u, 1u, 3u, 2u, 5u, 7u, 0u });
	auto list        = GpuList<GpuListType::Uint>();
	auto result      = GpuList<GpuListType::Uint>();
	auto indices     = GpuList<GpuListType::Uint>();
	auto indexResult = GpuList<GpuListType::Uint>();
	auto sortHelper  = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	result.resize(listData.size());
	indices.resize(indexData.size());
	indexResult.resize(indexData.size());
	sortHelper.resize(ListManipulation::sort_calculateNeededHelperListLength(list.length()));
	fillList(list.getWriteBuffer(), listData);
	fillList(indices.getWriteBuffer(), indexData);
	ListManipulation::sort(list.getWriteBuffer(), indices.getWriteBuffer(), sortHelper.getWriteBuffer(), list.length(), result.getWriteBuffer(), indexResult.getWriteBuffer());
	auto pass = validateList(result.getReadBuffer(), expectedResult, "sort (small values)");
	return pass && validateList(indexResult.getReadBuffer(), expectedIndexResult, "sort indices (small values)");
}

bool pbd::test::sortManySmallValues()
{
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
	
	auto list        = GpuList<GpuListType::Uint>();
	auto result      = GpuList<GpuListType::Uint>();
	auto indices     = GpuList<GpuListType::Uint>();
	auto indexResult = GpuList<GpuListType::Uint>();
	auto sortHelper  = GpuList<GpuListType::Uint>();
	list.resize(listData.size());
	result.resize(listData.size());
	indices.resize(indexData.size());
	indexResult.resize(indexData.size());
	sortHelper.resize(ListManipulation::sort_calculateNeededHelperListLength(list.length()));
	fillList(list.getWriteBuffer(), listData);
	fillList(indices.getWriteBuffer(), indexData);
	ListManipulation::sort(list.getWriteBuffer(), indices.getWriteBuffer(), sortHelper.getWriteBuffer(), list.length(), result.getWriteBuffer(), indexResult.getWriteBuffer());
	std::stable_sort(indexData.begin(), indexData.end(), [&listData](size_t i0, size_t i1) { return listData[i0] < listData[i1]; });
	std::sort(listData.begin(), listData.end());
	auto pass = validateList(result.getReadBuffer(), listData, "sort (many small values)");
	return pass && validateList(indexResult.getReadBuffer(), indexData, "sort indices (many small values)");
}

bool pbd::test::sortByPositions() // TODO fix test case
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

bool pbd::test::deleteThese()
{
	auto hiddenValues = std::vector<uint32_t>({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 });
	auto expectedResult = std::vector<uint32_t>({ 12, 13 });
	auto listA = IndexedList<GpuList<GpuListType::Uint>>(13);
	listA.increaseLength(8);
	auto listB = listA;
	auto listC = listA;
	listB.increaseLength(3);
	listC.increaseLength(2);
	fillList(listA.getHiddenList().getWriteBuffer(), hiddenValues);
	listB.deleteThese();
	auto pass = listA.length() == 0 && listB.length() == 0 && listC.length() == 2 && listA.getHiddenList().length() == 2;
	pass = validateList(listC.getHiddenList().getReadBuffer(), expectedResult, "IndexedList::deleteThese") && pass;
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