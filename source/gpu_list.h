#pragma once
#include "shader_provider.h"
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

		void set_length(size_t aLength);
		void request_length(size_t aLength);
		avk::buffer& length() const; // TODO becomes outdated if a subsequent call of write_buffer() causes duplication!
		const shader_provider::changing_length changing_length();
		void apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource) override;

		gpu_list& operator=(const gpu_list& aRhs);
		gpu_list& operator+=(const gpu_list& aRhs);
		gpu_list operator+(const gpu_list& aRhs) const;

		void set_owner(list_interface<gpu_list<4ui64>>* aOwner);
		const avk::buffer& read_buffer() const;
		/// <summary><para>Request a buffer for writing. The buffer is valid until another function of this object is called, or until this object is copied.</para></summary>
		avk::buffer& write_buffer();

		static void cleanup();

	private:
		void prepare_for_edit(size_t aNeededLength, bool aCurrentContentNeeded = false);
		void copy_list(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset = 0, size_t aTargetOffset = 0);
		void copy_bytes(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset = 0, size_t aTargetOffset = 0);

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
inline void pbd::gpu_list<Stride>::set_length(size_t aLength)
{
	if (mRequestedLength < aLength) mRequestedLength = aLength;

	prepare_for_edit(mRequestedLength, true);
	auto l = static_cast<uint32_t>(aLength);
	length()->fill(&l, 0, avk::sync::with_barriers_into_existing_command_buffer(shader_provider::cmd_bfr(), {}, {}));
	shader_provider::sync_after_transfer();
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::request_length(size_t aLength)
{
	mRequestedLength = aLength;
}

template<size_t Stride>
inline avk::buffer& pbd::gpu_list<Stride>::length() const
{
	return mData->mLengthAIsActive ? mData->mLengthA : mData->mLengthB;
}

template<size_t Stride>
inline const shader_provider::changing_length pbd::gpu_list<Stride>::changing_length()
{
	mData->mLengthAIsActive = !mData->mLengthAIsActive;
	return mData->mLengthAIsActive ? shader_provider::changing_length{ mData->mLengthB, mData->mLengthA } : shader_provider::changing_length{ mData->mLengthA, mData->mLengthB };
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource)
{
	if (mOwner != aEditSource && mOwner != nullptr) mOwner->apply_edit(aEditList, this);

	auto oldData = mData;
	prepare_for_edit(mRequestedLength);
	shader_provider::copy_scattered_read(oldData->mBuffer, mData->mBuffer, aEditList.read_buffer(), aEditList.length(), length(), Stride);
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::set_owner(list_interface<gpu_list<4ui64>>* aOwner)
{
	mOwner = aOwner;
}

template<size_t Stride>
inline const avk::buffer& pbd::gpu_list<Stride>::read_buffer() const
{
	return mData->mBuffer;
}

template<size_t Stride>
inline avk::buffer& pbd::gpu_list<Stride>::write_buffer()
{
	prepare_for_edit(mRequestedLength, true);
	return mData->mBuffer;
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

	prepare_for_edit(mRequestedLength, true);
	shader_provider::append_list(mData->mBuffer, aRhs.mData->mBuffer, changing_length(), aRhs.length(), Stride);
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
	auto currentCapacity = mData == nullptr ? 0ui64 : mData->mBuffer->meta<avk::storage_buffer_meta>().total_size() / Stride;

	if (currentCapacity   == 0 && aNeededLength   ==             0) return;
	if (mData.use_count() == 2 && currentCapacity >= aNeededLength) return;
	if (aNeededLength == 0) { mData = nullptr; return; }

	auto oldData = mData;
	mData = get_list(aNeededLength);

	if (oldData == nullptr) {
		set_length(0);
	} else {
		copy_bytes(oldData->mLengthAIsActive ? oldData->mLengthA : oldData->mLengthB, length(), 4);
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
	copy_bytes(aSource, aTarget, aCopiedLength * Stride, aSourceOffset * Stride, aTargetOffset * Stride);
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::copy_bytes(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset, size_t aTargetOffset)
{
	if (aCopiedLength == 0) return;

	assert(aSource->meta<avk::storage_buffer_meta>().total_size() >= aSourceOffset + aCopiedLength);
	assert(aTarget->meta<avk::storage_buffer_meta>().total_size() >= aTargetOffset + aCopiedLength);

	auto copyRegion = vk::BufferCopy{}
		.setSrcOffset(aSourceOffset)
		.setDstOffset(aTargetOffset)
		.setSize(aCopiedLength);
	shader_provider::cmd_bfr()->handle().copyBuffer(aSource->buffer_handle(), aTarget->buffer_handle(), { copyRegion });
	shader_provider::sync_after_transfer();
}

template<size_t Stride>
inline std::shared_ptr<typename pbd::gpu_list<Stride>::gpu_list_data> pbd::gpu_list<Stride>::get_list(size_t aMinLength)
{
	auto minSize = aMinLength * Stride;
	std::shared_ptr<gpu_list_data>* bestExisting = nullptr;
	for (auto& data : mReservedLists)
	{
		if (data.use_count() == 1 && data->mBuffer->meta<avk::storage_buffer_meta>().total_size() >= minSize)
		{
			if (bestExisting == nullptr || (*bestExisting)->mBuffer->meta<avk::storage_buffer_meta>().total_size() > data->mBuffer->meta<avk::storage_buffer_meta>().total_size())
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
