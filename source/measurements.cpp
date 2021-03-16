#include "measurements.h"
#include "shader_provider.h"

std::unordered_map<std::string, std::tuple<vk::UniqueQueryPool, std::array<uint32_t, 2>, float, uint32_t>> measurements::mIntervals;
std::unordered_map<std::string, avk::buffer> measurements::mBuffers;

uint32_t measurements::async_read_uint(const std::string& aName, const avk::buffer& aBuffer)
{
	auto iter = mBuffers.find(aName);
	if (iter == mBuffers.end()) {
		iter = mBuffers.try_emplace(aName).first;
		iter->second = gvk::context().create_buffer(
			avk::memory_usage::host_coherent, vk::BufferUsageFlagBits::eTransferDst,
			avk::generic_buffer_meta::create_from_size(4)
		);
	}
	auto copyRegion = vk::BufferCopy{}
		.setSrcOffset(0)
		.setDstOffset(0)
		.setSize(4);
	shader_provider::start_recording();
	shader_provider::cmd_bfr()->handle().copyBuffer(aBuffer->handle(), iter->second->handle(), { copyRegion });
	shader_provider::end_recording();
	return iter->second->read<uint32_t>(0, avk::sync::not_required()); // read old value from buffer
}

void measurements::record_timing_interval_start(const std::string& aName)
{
	auto& queryPool = add_timing_interval_and_get_query_pool(aName);
	auto query = static_cast<uint32_t>(gvk::context().main_window()->current_in_flight_index()) * 2u;
	shader_provider::cmd_bfr()->handle().resetQueryPool(queryPool, query, 2u);
	shader_provider::cmd_bfr()->handle().writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, query);
}

void measurements::record_timing_interval_end(const std::string& aName)
{
	auto& queryPool = add_timing_interval_and_get_query_pool(aName);
	auto query = static_cast<uint32_t>(gvk::context().main_window()->current_in_flight_index()) * 2u + 1u;
	shader_provider::cmd_bfr()->handle().writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, query);
}

float measurements::get_timing_interval_in_ms(const std::string& aName)
{
	auto iter = mIntervals.find(aName);
	if (iter == mIntervals.end()) {
		return 0.0f;
	}
	auto& [queryPool, timestamps, avgRendertime, queryUsed] = iter->second;
	auto mainWindow = gvk::context().main_window();
	auto oldestFrame = std::max(0i64, mainWindow->current_frame() - mainWindow->number_of_frames_in_flight() + 1i64);
	auto oldestInFlight = static_cast<uint32_t>(mainWindow->in_flight_index_for_frame(oldestFrame));
	if (!((queryUsed >> oldestInFlight) & 1u)) {
		return 0.0f;
	}
	auto query = oldestInFlight * 2u; // choose oldest query
	gvk::context().mLogicalDevice.getQueryPoolResults(*queryPool, query, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint32_t), vk::QueryResultFlagBits::eWait);
	float delta = (timestamps[1] - timestamps[0]) * gvk::context().physical_device().getProperties().limits.timestampPeriod / 1000000.0f;
	avgRendertime = avgRendertime * 0.9f + delta * 0.1f;
	return avgRendertime;
}

void measurements::clean_up_resources()
{
	mIntervals.clear();
	mBuffers.clear();
}

void measurements::debug_label_start(const std::string& aName, const glm::vec4& aColor)
{
	auto color = std::array<float, 4>({ aColor[0], aColor[1], aColor[2], aColor[3] });
	auto label = vk::DebugUtilsLabelEXT{}.setPLabelName(aName.c_str()).setColor(color);
	shader_provider::cmd_bfr()->handle().beginDebugUtilsLabelEXT(label, gvk::context().dynamic_dispatch());
}

void measurements::debug_label_end()
{
	shader_provider::cmd_bfr()->handle().endDebugUtilsLabelEXT(gvk::context().dynamic_dispatch());
}

vk::QueryPool& measurements::add_timing_interval_and_get_query_pool(const std::string& aName)
{
	auto iter = mIntervals.find(aName);
	if (iter == mIntervals.end()) {
		vk::QueryPoolCreateInfo queryPoolCreateInfo;
		queryPoolCreateInfo.setQueryCount(static_cast<uint32_t>(gvk::context().main_window()->number_of_frames_in_flight()) * 2u);
		queryPoolCreateInfo.setQueryType(vk::QueryType::eTimestamp);

		iter = mIntervals.try_emplace(aName, gvk::context().mLogicalDevice.createQueryPoolUnique(queryPoolCreateInfo), std::array<uint32_t, 2>(), 0.0f, 0u).first;
	}
	auto& queryUsed = std::get<3>(iter->second);
	queryUsed = queryUsed | 1u << gvk::context().main_window()->current_in_flight_index();
	return *std::get<0>(iter->second);
}
