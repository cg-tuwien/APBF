#pragma once

#include <gvk.hpp>

class measurements
{
public:
	measurements() = delete;

	static void record_timing_interval_start(const std::string& aName);
	static void record_timing_interval_end(const std::string& aName);
	// request last timing interval from GPU and return averaged interval from previous measurements (in ms)
	static float get_timing_interval_in_ms(const std::string& aName);
	static void clean_up_timing_resources();

private:
	static vk::QueryPool& add_timing_interval_and_get_query_pool(const std::string& aName);
	static std::unordered_map<std::string, std::tuple<vk::UniqueQueryPool, std::array<uint32_t, 2>, float>> mIntervals;
};