#pragma once

#include <gvk.hpp>

class measurements
{
public:
	measurements() = delete;
	static uint32_t async_read_uint(const std::string& aName, const avk::buffer& aBuffer);
	static void record_timing_interval_start(const std::string& aName);
	static void record_timing_interval_end(const std::string& aName);
	// read oldest timing interval from GPU and return averaged interval from previous measurements (in ms)
	static float get_timing_interval_in_ms(const std::string& aName);
	static void clean_up_resources();

	static void debug_label_start(const std::string& aName, const glm::vec4& aColor = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
	static void debug_label_end();

private:
	static vk::QueryPool& add_timing_interval_and_get_query_pool(const std::string& aName);
	static std::unordered_map<std::string, std::tuple<vk::UniqueQueryPool, std::array<uint32_t, 2>, float, uint32_t>> mIntervals;
	static std::unordered_map<std::string, avk::buffer> mBuffers;
};
