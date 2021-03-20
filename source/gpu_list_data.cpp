#include "gpu_list_data.h"

std::list<pbd::gpu_list_data::list_entry> pbd::gpu_list_data::mReservedLists;
uint32_t pbd::gpu_list_data::mGarbageCollectionCountBeforeDeletion = 60u;

std::shared_ptr<pbd::gpu_list_data> pbd::gpu_list_data::get_list(size_t aMinLength, uint32_t aStride)
{
	auto minSize = aMinLength * aStride;
	list_entry* bestExisting = nullptr;
	for (auto& data : mReservedLists)
	{
		if (data.mGpuListData.use_count() == 1 && data.mGpuListData->size() >= minSize)
		{
			if (bestExisting == nullptr || bestExisting->mGpuListData->size() > data.mGpuListData->size())
			{
				bestExisting = &data;
			}
		}
	}

//	if (bestExisting != nullptr && bestExisting->mGpuListData->size() > std::max(4096ull, static_cast<size_t>(minSize * 1.5))) {
//		bestExisting = nullptr;
//	}

	if (bestExisting == nullptr) {

		auto newBuffer = gvk::context().create_buffer(
			avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
			avk::storage_buffer_meta::create_from_size(aMinLength * aStride),
			avk::instance_buffer_meta::create_from_element_size(aStride, aMinLength) // TODO should this be avoided?
		);
		auto newLengthBuffer = gvk::context().create_buffer(
			avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
			avk::storage_buffer_meta::create_from_size(4)
		);
		mReservedLists.push_back({ std::make_shared<gpu_list_data>(std::move(newBuffer), std::move(newLengthBuffer)), 0 });
		if (mReservedLists.size() >= 50)
		{
			LOG_WARNING("high number of reserved GPU buffers (" + std::to_string(mReservedLists.size()) + ")");
		}
		bestExisting = &mReservedLists.back();
	}
	bestExisting->mTimeUnused = 0u;
	return bestExisting->mGpuListData;
}

void pbd::gpu_list_data::garbage_collection()
{
	// TODO find out why this sometimes breaks everything
//	mReservedLists.remove_if([](auto& data) { return data.mGpuListData.use_count() == 1 && ++data.mTimeUnused >= mGarbageCollectionCountBeforeDeletion; });
}

void pbd::gpu_list_data::cleanup()
{
	mReservedLists.clear();
}

size_t pbd::gpu_list_data::size()
{
	return mBuffer->meta_at_index<avk::buffer_meta>().total_size();
}
