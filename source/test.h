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

		template <class T>
		static pbd::gpu_list<sizeof(T)> to_gpu_list(const std::vector<T>& aData);
		static bool validate_length(const avk::buffer& aLength, size_t aExpectedLength, const std::string& aTestName);
		template <class T>
		static bool validate_list(const avk::buffer& aList, const std::vector<T>& aExpectedData, const std::string& aTestName, uint32_t aOffset = 0u);
		template <class T>
		static bool approximately_equal(T v1, T v2);

		static bool gpu_list_concatenation_1();
		static bool gpu_list_concatenation_2();
		static bool gpu_list_append_empty();
		static bool gpu_list_apply_edit();
		static bool indexed_list_write_increasing_sequence();
		static bool indexed_list_apply_hidden_edit_1();
		static bool indexed_list_apply_hidden_edit_2();
		static bool indexed_list_apply_hidden_edit_3();
		static bool prefix_sum();
		static bool long_prefix_sum();
		static bool very_long_prefix_sum();
		static bool short_prefix_sum_in_long_buffer();
		static bool sort();
		static bool sort_many_values();
		static bool sort_small_values();
		static bool sort_many_small_values();
		static bool sort_few_values_in_long_buffer();
		static bool delete_these_1();
		static bool delete_these_2();
		static bool delete_these_3();
		static bool duplicate_these();
		static bool duplicate_these_empty();
		static bool neighborhood_brute_force();
		static bool neighborhood_green();
		static bool time_machine();
/*		static bool sortByPositions();
		static bool merge();
		static bool mergeGenerator();
		static bool mergeGeneratorGrid();*/
	};




	template<class T>
	inline pbd::gpu_list<sizeof(T)> test::to_gpu_list(const std::vector<T>& aData)
	{
		auto result = pbd::gpu_list<sizeof(T)>();
		result.request_length(1).set_length(aData.size());
		algorithms::copy_bytes(aData.data(), result.write().buffer(), aData.size() * sizeof(T));
		return result;
	}

	template<class T>
	inline bool test::validate_list(const avk::buffer& aList, const std::vector<T>& aExpectedData, const std::string& aTestName, uint32_t aOffset)
	{
		auto data = std::vector<T>();
		data.resize(aList->meta_at_index<avk::buffer_meta>().total_size() / sizeof(T));
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
