#include "measurements.h"
#include "shader_provider.h"

std::unordered_map<std::string, std::tuple<vk::UniqueQueryPool, std::array<uint32_t, 2>, float>> measurements::mIntervals;

void measurements::record_timing_interval_start(const std::string& aName)
{
	auto& queryPool = add_timing_interval_and_get_query_pool(aName);
	auto query = gvk::context().main_window()->current_in_flight_index() * 2u;
	shader_provider::cmd_bfr()->handle().resetQueryPool(queryPool, query, 2u);
	shader_provider::cmd_bfr()->handle().writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, query);
}

void measurements::record_timing_interval_end(const std::string& aName)
{
	auto& queryPool = add_timing_interval_and_get_query_pool(aName);
	auto query = gvk::context().main_window()->current_in_flight_index() * 2u + 1u;
	shader_provider::cmd_bfr()->handle().writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, query);
}

float measurements::get_timing_interval_in_ms(const std::string& aName)
{
	auto iter = mIntervals.find(aName);
	if (iter == mIntervals.end()) {
		return 0.0f;
	}
	auto& [queryPool, timestamps, avgRendertime] = iter->second;
	auto query = gvk::context().main_window()->current_in_flight_index() * 2u;
	gvk::context().mLogicalDevice.getQueryPoolResults(*queryPool, query, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint32_t), vk::QueryResultFlagBits::eWait);
	float delta = (timestamps[1] - timestamps[0]) * gvk::context().physical_device().getProperties().limits.timestampPeriod / 1000000.0f;
	avgRendertime = avgRendertime * 0.9f + delta * 0.1f;
	return avgRendertime;
}

void measurements::clean_up_timing_resources()
{
	mIntervals.clear();
}

vk::QueryPool& measurements::add_timing_interval_and_get_query_pool(const std::string& aName)
{
	auto iter = mIntervals.find(aName);
	if (iter == mIntervals.end()) {
		vk::QueryPoolCreateInfo queryPoolCreateInfo;
		queryPoolCreateInfo.setQueryCount(gvk::context().main_window()->number_of_frames_in_flight() * 2u);
		queryPoolCreateInfo.setQueryType(vk::QueryType::eTimestamp);

		iter = mIntervals.try_emplace(aName, gvk::context().mLogicalDevice.createQueryPoolUnique(queryPoolCreateInfo), std::array<uint32_t, 2>(), 0.0f).first;
	}
	return *std::get<0>(iter->second);
}
