#pragma once
#include "algorithms.h"
#include "list_interface.h"

namespace pbd
{
	/// <summary><para>A list with its elements stored only in GPU memory.</para>
	/// <para>Copying this class is cheap as it does not immediately duplicate the GPU memory (lazy copy).</para>
	/// <para>Use <seealso cref="gpu_list::read_buffer"/> for read-access to the GPU memory, but use <seealso cref="gpu_list::write_buffer"/> if you need write-access.</para></summary>
	template<size_t Stride>
	class gpu_list :
		public list_interface<pbd::gpu_list<4ui64>>
	{
	public:
		gpu_list();
		gpu_list(const gpu_list& aGpuList);
		~gpu_list() = default;

		// empty() does not mean "no elements" (the CPU-side wouldn't know that), but "no GPU buffer yet assigned"
		// It only returns true if the list was not copied from a non-empty list and write() was never called for it
		bool empty() const;
		avk::buffer& length() const;
		gpu_list& set_length(size_t aLength);
		gpu_list& set_length(const avk::buffer& aLength);
		gpu_list& request_length(size_t aLength);
		size_t requested_length();
		void apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource) override;

		gpu_list& operator=(const gpu_list& aRhs);
		gpu_list& operator+=(const gpu_list& aRhs);
		gpu_list operator+(const gpu_list& aRhs) const;

		gpu_list& set_owner(list_interface<gpu_list<4ui64>>* aOwner);
		/// <summary><para>Request the buffer containing the list. The buffer is valid until another function of this object is called, or until this object is copied.</para></summary>
		avk::buffer& buffer() const;

		/// <summary><para>Request write access to the list. The intended use is: gpu_list.write().buffer() and gpu_list.write().length().</para></summary>
		gpu_list& write();

		static void cleanup();

	private:
		void prepare_for_edit(size_t aNeededLength, bool aCurrentContentNeeded = false);
		void copy_list(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset = 0, size_t aTargetOffset = 0);

		class gpu_list_data
		{
		public:
			gpu_list_data(avk::buffer aBuffer, avk::buffer aLengthA, avk::buffer aLengthB) : mBuffer{ std::move(aBuffer) }, mLengthA{ std::move(aLengthA) }, mLengthB{ std::move(aLengthB) }, mLengthAIsActive{ true } {}
			avk::buffer mBuffer;
			avk::buffer mLengthA;
			avk::buffer mLengthB;
			bool mLengthAIsActive;
		};

		std::shared_ptr<gpu_list_data> mData;
		size_t mRequestedLength;
		list_interface<gpu_list<4ui64>>* mOwner = nullptr;

		// static buffer cache
		static std::shared_ptr<gpu_list_data> get_list(size_t aMinLength);
		static std::list<std::shared_ptr<gpu_list_data>> mReservedLists;
	};
}









template<size_t Stride>
std::list<std::shared_ptr<typename pbd::gpu_list<Stride>::gpu_list_data>> pbd::gpu_list<Stride>::mReservedLists;

template<size_t Stride>
inline pbd::gpu_list<Stride>::gpu_list()
{
	mRequestedLength = 0ui64;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>::gpu_list(const gpu_list& aGpuList)
{
	mData = aGpuList.mData;
	mRequestedLength = aGpuList.mRequestedLength;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::set_length(size_t aLength)
{
	if (mRequestedLength < aLength) mRequestedLength = aLength;

	auto l = static_cast<uint32_t>(aLength);
	algorithms::copy_bytes(&l, write().length(), 4);
	return *this;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::set_length(const avk::buffer& aLength)
{
	algorithms::copy_bytes(aLength, write().length(), 4);
	return *this;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::request_length(size_t aLength)
{
	mRequestedLength = aLength;
	return *this;
}

template<size_t Stride>
inline size_t pbd::gpu_list<Stride>::requested_length()
{
	return mRequestedLength;
}

template<size_t Stride>
inline bool pbd::gpu_list<Stride>::empty() const
{
	return mData == nullptr;
}

template<size_t Stride>
inline avk::buffer& pbd::gpu_list<Stride>::length() const
{
	return mData->mLengthAIsActive ? mData->mLengthA : mData->mLengthB;
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource)
{
	if (mOwner != aEditSource && mOwner != nullptr) mOwner->apply_edit(aEditList, this);

	auto oldData = mData;
	prepare_for_edit(mRequestedLength);
	shader_provider::copy_scattered_read(oldData->mBuffer, mData->mBuffer, aEditList.buffer(), aEditList.length(), Stride);
	set_length(aEditList.length());
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::set_owner(list_interface<gpu_list<4ui64>>* aOwner)
{
	mOwner = aOwner;
	return *this;
}

template<size_t Stride>
inline avk::buffer& pbd::gpu_list<Stride>::buffer() const
{
	return mData->mBuffer;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::write()
{
	prepare_for_edit(mRequestedLength, true);
	return *this;
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::cleanup()
{
	mReservedLists.clear();
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::operator=(const gpu_list& aRhs)
{
	mData = aRhs.mData;
	mRequestedLength = aRhs.mRequestedLength;
	return *this;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::operator+=(const gpu_list& aRhs)
{
	if (mData == nullptr)
	{
		mData = aRhs.mData;
		return *this;
	}

	if (aRhs.mData == nullptr) return *this;

	set_length(shader_provider::append_list(write().buffer(), aRhs.mData->mBuffer, write().length(), aRhs.length(), Stride));
	return *this;
}

template<size_t Stride>
inline pbd::gpu_list<Stride> pbd::gpu_list<Stride>::operator+(const gpu_list& aRhs) const
{
	auto result = *this;
	result.mRequestedLength = std::max(mRequestedLength, aRhs.mRequestedLength);
	result += aRhs;
	return result;
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::prepare_for_edit(size_t aNeededLength, bool aCurrentContentNeeded)
{
	auto currentCapacity = mData == nullptr ? 0ui64 : mData->mBuffer->meta_at_index<avk::buffer_meta>().total_size() / Stride;

	if (currentCapacity   == 0 && aNeededLength   ==             0) return;
	if (mData.use_count() == 2 && currentCapacity >= aNeededLength) return;
	if (aNeededLength == 0) { mData = nullptr; return; }

	auto oldData = mData;
	mData = get_list(aNeededLength);

	if (oldData == nullptr) {
		set_length(0);
	} else {
		algorithms::copy_bytes(oldData->mLengthAIsActive ? oldData->mLengthA : oldData->mLengthB, length(), 4);
	}

	if (aCurrentContentNeeded)
	{
		auto copiedLength = std::min(currentCapacity, aNeededLength);
		copy_list(oldData->mBuffer, mData->mBuffer, copiedLength);
	}
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::copy_list(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset, size_t aTargetOffset)
{
	algorithms::copy_bytes(aSource, aTarget, aCopiedLength * Stride, aSourceOffset * Stride, aTargetOffset * Stride);
}

template<size_t Stride>
inline std::shared_ptr<typename pbd::gpu_list<Stride>::gpu_list_data> pbd::gpu_list<Stride>::get_list(size_t aMinLength)
{
	auto minSize = aMinLength * Stride;
	std::shared_ptr<gpu_list_data>* bestExisting = nullptr;
	for (auto& data : mReservedLists)
	{
		if (data.use_count() == 1 && data->mBuffer->meta_at_index<avk::buffer_meta>().total_size() >= minSize)
		{
			if (bestExisting == nullptr || (*bestExisting)->mBuffer->meta_at_index<avk::buffer_meta>().total_size() > data->mBuffer->meta_at_index<avk::buffer_meta>().total_size())
			{
				bestExisting = &data;
			}
		}
	}
	if (bestExisting == nullptr) {

		auto newBuffer = gvk::context().create_buffer(
			avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
			avk::storage_buffer_meta::create_from_size(aMinLength * Stride),
			avk::instance_buffer_meta::create_from_element_size(Stride, aMinLength) // TODO should this be avoided?
		);
		auto newLengthABuffer = gvk::context().create_buffer(
			avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
			avk::storage_buffer_meta::create_from_size(4)
		);
		auto newLengthBBuffer = gvk::context().create_buffer(
			avk::memory_usage::device, vk::BufferUsageFlagBits::eTransferSrc,
			avk::storage_buffer_meta::create_from_size(4)
		);
		mReservedLists.push_back(std::make_shared<gpu_list_data>(std::move(newBuffer), std::move(newLengthABuffer), std::move(newLengthBBuffer)));
		if (mReservedLists.size() >= 50)
		{
			LOG_DEBUG("high number of reserved GPU buffers (" + std::to_string(mReservedLists.size()) + ")");
		}
		bestExisting = &mReservedLists.back();
	}
	return *bestExisting;
}
