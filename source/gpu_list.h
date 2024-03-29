#pragma once
#include "gpu_list_data.h"
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
		/// <summary><para>Read the list content from the GPU. Only intended for debugging.</para></summary>
		template<class T>
		std::vector<T> read(bool aBeyondLength = false) const;

		/// <summary><para>Request write access to the list. The intended use is: gpu_list.write().buffer() and gpu_list.write().length().</para></summary>
		gpu_list& write();

		// only defined for gpu_list<4>; interprets the values as uint and sorts them into ascending order
		void sort(size_t aValueUpperBound = MAXUINT32);

		template<size_t NewStride>
		gpu_list<NewStride> convert_to_stride() const;

	private:
		void prepare_for_edit(size_t aNeededLength, bool aCurrentContentNeeded = false);
		void copy_list(const avk::buffer& aSource, const avk::buffer& aTarget, size_t aCopiedLength, size_t aSourceOffset = 0, size_t aTargetOffset = 0);

		std::shared_ptr<gpu_list_data> mData;
		size_t mRequestedLength;
		list_interface<gpu_list<4ui64>>* mOwner = nullptr;
	};
}









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
	if (aLength == 0 && mData.use_count() > 2) mData = nullptr;

	auto l = static_cast<uint32_t>(aLength);
	algorithms::copy_bytes(&l, write().length(), 4);
	return *this;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::set_length(const avk::buffer& aLength)
{
	// TODO limit to requested length? (prepare_for_edit() is always called with the requested length)
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
	return mData->mLength;
}

template<size_t Stride>
inline void pbd::gpu_list<Stride>::apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource)
{
	if (mOwner != aEditSource && mOwner != nullptr) mOwner->apply_edit(aEditList, this);
	if (this == aEditSource) return;

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
	// if this assert fails, you have either forgotten to call write() or have not requested a length
	assert(mData != nullptr);
	return mData->mBuffer;
}

template<size_t Stride>
template<class T>
inline std::vector<T> pbd::gpu_list<Stride>::read(bool aBeyondLength) const
{
	auto data = std::vector<T>();
	if (empty()) return data;

	auto wasRecording = shader_provider::is_recording();
	if (wasRecording) shader_provider::end_recording();
	auto len = 0u;
	data.resize(buffer()->meta_at_index<avk::buffer_meta>().total_size() / sizeof(T));
	buffer()->read(data.data(), 0, avk::sync::wait_idle(true));
	if (!aBeyondLength) {
		length()->read(&len, 0, avk::sync::wait_idle(true));
		data.resize(len);
	}
	if (wasRecording) shader_provider::start_recording();
	return data;
}

template<size_t Stride>
inline pbd::gpu_list<Stride>& pbd::gpu_list<Stride>::write()
{
	prepare_for_edit(mRequestedLength, true);
	return *this;
}

template<>
inline void pbd::gpu_list<4>::sort(size_t aValueUpperBound)
{
	auto  unsortedList      = *this;
	auto& sortedList        = *this;
	auto  unsortedIndexList = pbd::gpu_list<4>().request_length(requested_length());
	auto  sortedIndexList   = pbd::gpu_list<4>().request_length(requested_length());
	auto  sortHelper        = pbd::gpu_list<4>().request_length(pbd::algorithms::sort_calculate_needed_helper_list_length(requested_length()));

	sortedIndexList.set_length(length());

	shader_provider::write_sequence(unsortedIndexList.write().buffer(), length(), 0u, 1u);
	pbd::algorithms::sort(unsortedList.write().buffer(), unsortedIndexList.write().buffer(), sortHelper.write().buffer(), length(), unsortedList.requested_length(), sortedList.write().buffer(), sortedIndexList.write().buffer(), static_cast<uint32_t>(aValueUpperBound));
	apply_edit(sortedIndexList, this);
}

template<size_t Stride>
template<size_t NewStride>
inline pbd::gpu_list<NewStride> pbd::gpu_list<Stride>::convert_to_stride() const
{
	auto result = gpu_list<NewStride>().request_length(requested_length());
	result.set_length(length());
	shader_provider::copy_with_differing_stride(buffer(), result.write().buffer(), length(), Stride, NewStride);
	return result;
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

	// TODO limit length to requested length instead of actual buffer length
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
	mData = gpu_list_data::get_list(aNeededLength, Stride);

	if (oldData == nullptr) {
		set_length(0);
	} else {
		algorithms::copy_bytes(oldData->mLength, length(), 4);
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
