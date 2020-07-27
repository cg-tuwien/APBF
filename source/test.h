#pragma once
#include <gvk.hpp>

namespace pbd
{
	class test
	{
	public:
		test() = delete;

		static void test_all();
		static void test_quick();

		template <class T>
		static bool validate_list(const avk::buffer& pList, const std::vector<T>& pExpectedData, const std::string& pTestName);
		template <class T>
		static bool approximately_equal(T v1, T v2);

		static bool gpu_list_concatenation();
		static bool gpu_list_concatenation2();
		static bool gpu_list_apply_edit();
/*		static bool indexedList_writeDecreasingSequence();
		static bool indexedList_applyHiddenEdit();
		static bool indexedList_applyHiddenEdit2();
		static bool indexedList_applyHiddenEdit3();
		static bool prefixSum();
		static bool longPrefixSum();
		static bool veryLongPrefixSum();
		static bool sort();
		static bool sortManyValues();
		static bool sortSmallValues();
		static bool sortManySmallValues();
		static bool sortByPositions();
		static bool deleteThese();
		static bool merge();
		static bool mergeGenerator();
		static bool mergeGeneratorGrid();*/
	};




	template<class T>
	inline bool test::validate_list(const avk::buffer& pList, const std::vector<T>& pExpectedData, const std::string& pTestName)
	{
		auto data = std::vector<T>();
		data.resize(pExpectedData.size());
		pList->read(data.data(), 0, avk::sync::wait_idle(true));
		/*auto pData = pList->map(Buffer::MapType::Read);
		for (auto i = 0u; i < expectedData.size(); i++)
		{
			auto address = static_cast<void*>(static_cast<char*>(pData) + i * pList->getElementSize()); // hopefully char* is always 1 byte
			data.push_back(*static_cast<T*>(address));
		}
		pList->unmap();*/

		for (auto i = 0u; i < pExpectedData.size(); i++)
		{
			if (!approximately_equal(pExpectedData[i], data[i]))
			{
				LOG_WARNING("TEST FAIL: [" + pTestName + "] - at list index " + std::to_string(i));
				return false;
			}
		}
		return true;
	}

	template<class T>
	inline bool test::approximately_equal(T v1, T v2)
	{
		return v1 == v2;
	}

	template<>
	inline bool test::approximately_equal(float v1, float v2)
	{
		return glm::epsilonEqual(v1, v2, 0.000005f);
	}

	template<>
	inline bool test::approximately_equal(glm::vec3 v1, glm::vec3 v2)
	{
		return glm::all(glm::epsilonEqual(v1, v2, glm::vec3(0.000005f)));
	}
}
