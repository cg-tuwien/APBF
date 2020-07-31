#pragma once
#include <gvk.hpp>
#include "list_definitions.h"

namespace pbd
{
	class test
	{
	public:
		test() = delete;

		static void test_all();
		static void test_quick();

		template <class T>
		static pbd::gpu_list<sizeof(T)> to_gpu_list(const std::vector<T>& aData);
		static bool validate_length(const avk::buffer& aLength, size_t aExpectedLength, const std::string& aTestName);
		template <class T>
		static bool validate_list(const avk::buffer& aList, const std::vector<T>& aExpectedData, const std::string& aTestName, uint32_t aOffset = 0u);
		template <class T>
		static bool approximately_equal(T v1, T v2);

		static bool gpu_list_concatenation();
		static bool gpu_list_concatenation2();
		static bool gpu_list_apply_edit();
		static bool indexed_list_write_increasing_sequence();
//		static bool indexedList_applyHiddenEdit();
//		static bool indexedList_applyHiddenEdit2();
//		static bool indexedList_applyHiddenEdit3();
		static bool prefix_sum();
		static bool long_prefix_sum();
		static bool very_long_prefix_sum();
		static bool sort();
//		static bool sortManyValues();
		static bool sort_small_values();
/*		static bool sortManySmallValues();
		static bool sortByPositions();
		static bool deleteThese();
		static bool merge();
		static bool mergeGenerator();
		static bool mergeGeneratorGrid();*/
	};




	template<class T>
	inline pbd::gpu_list<sizeof(T)> test::to_gpu_list(const std::vector<T>& aData)
	{
		auto result = pbd::gpu_list<sizeof(T)>();
		result.set_length(aData.size());
		result.write().buffer()->fill(aData.data(), 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr()));
		return result;
	}

	template<class T>
	inline bool test::validate_list(const avk::buffer& aList, const std::vector<T>& aExpectedData, const std::string& aTestName, uint32_t aOffset)
	{
		auto data = std::vector<T>();
		data.resize(aList->meta<avk::storage_buffer_meta>().total_size() / sizeof(T));
		aList->read(data.data(), 0, avk::sync::wait_idle(true));

		for (auto i = 0u; i < aExpectedData.size(); i++)
		{
			if (!approximately_equal(aExpectedData[i], data[i + aOffset]))
			{
				LOG_WARNING("TEST FAIL: [" + aTestName + "] - at list index " + std::to_string(i + aOffset));
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
