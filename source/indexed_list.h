#pragma once

#include "gpu_list.h"

namespace pbd
{
	template<class... Data>
	class time_machine;

	/// <summary><para>A container for two lists: the index list and the hidden list. The index list entries are indices pointing into the hidden list.</para>
	/// <para>The hidden list is shared among all descendants of an indexed list object.</para>
	/// <para>List manipulation (set_length, add,...) targets the index list. increase_length() initializes new indices to point to currently unused hidden list entries.</para></summary>
	template<class DataList>
	class indexed_list :
		public list_interface<gpu_list<4ui64>>
	{
	public:
		indexed_list(size_t aAllocatedHiddenDataLength = 0);
		indexed_list(const indexed_list& aIndexedList);
		~indexed_list();

		indexed_list& share_hidden_data_from(const indexed_list& aBenefactor);
		void delete_these();
		indexed_list duplicate_these();

		bool empty() const;
		avk::buffer& length() const;
		indexed_list& set_length(size_t aLength);
		indexed_list& set_length(const avk::buffer& aLength);
		indexed_list& request_length(size_t aLength);
		size_t requested_length();
		indexed_list increase_length(size_t aAddedLength);
		void apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource) override;

		indexed_list& operator=(const indexed_list& aRhs);
		indexed_list& operator+=(const indexed_list& aRhs);
		indexed_list operator+(const indexed_list& aRhs) const;

		void set_owner(list_interface<gpu_list<4ui64>>* aOwner);
		/// <summary><para>Request the buffer containing the list. The buffer is valid until another function of this object is called, or until this object is copied.</para></summary>
		avk::buffer& index_buffer() const;
		DataList& hidden_list();
		/// <summary><para>Read the index list content from the GPU. Only intended for debugging.</para></summary>
		std::vector<uint32_t> index_read(bool aBeyondLength = false) const;

		/// <summary><para>Request write access to the list. The intended use is: indexed_list.write().index_buffer() and indexed_list.write().length().</para></summary>
		indexed_list& write();

		void sort(size_t aValueUpperBound);
		void sort();

	private:
		class hidden_data :
			public list_interface<gpu_list<4ui64>>
		{
		public:
			hidden_data() { mData.set_owner(this); }
			void add_owner(indexed_list* aOwner) { mOwners.push_back(aOwner); }
			void remove_owner(indexed_list* aOwner) { mOwners.remove(aOwner); }
			void apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource) override
			{
				for (auto aOwner : mOwners) if (aOwner != aEditSource) aOwner->apply_hidden_edit(aEditList);
				if (&mData != aEditSource) mData.apply_edit(aEditList, this);
			}

			DataList mData;
			std::list<indexed_list*> mOwners;
		};

		void apply_hidden_edit(gpu_list<4ui64>& aEditList);

		gpu_list<4ui64> mIndexList;
		std::shared_ptr<hidden_data> mHiddenData;
		list_interface<gpu_list<4ui64>>* mOwner = nullptr;
		bool mSorted;

		template<typename...>
		friend class pbd::time_machine;
	};
}









template<class DataList>
inline pbd::indexed_list<DataList>::indexed_list(size_t aAllocatedHiddenDataLength)
{
	mHiddenData = std::make_shared<hidden_data>();
	mHiddenData->mData.request_length(aAllocatedHiddenDataLength);
	mIndexList.set_owner(this);
	mHiddenData->add_owner(this);
	mSorted = true;
}

template<class DataList>
inline pbd::indexed_list<DataList>::indexed_list(const indexed_list& aIndexedList) :
	mIndexList(aIndexedList.mIndexList)
{
	mHiddenData = aIndexedList.mHiddenData;
	mIndexList.set_owner(this);
	mHiddenData->add_owner(this);
	mSorted = aIndexedList.mSorted;
}

template<class DataList>
inline pbd::indexed_list<DataList>::~indexed_list()
{
	mHiddenData->remove_owner(this);
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::share_hidden_data_from(const indexed_list& aBenefactor)
{
	mHiddenData->remove_owner(this);
	mHiddenData = aBenefactor.mHiddenData;
	mHiddenData->add_owner(this);
	return *this;
}

template<class DataList>
inline void pbd::indexed_list<DataList>::delete_these()
{
	auto helperListLength = mHiddenData->mData.requested_length();
	auto helperList   = gpu_list<4ui64>().request_length(helperListLength);
	auto prefixHelper = gpu_list<4ui64>().request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(helperListLength));
	shader_provider::write_sequence(helperList.write().buffer(), mHiddenData->mData.length(), 1u, 0u);
	shader_provider::scattered_write(mIndexList.buffer(), helperList.write().buffer(), mIndexList.length(), 0u);
	mIndexList.set_length(0);
	algorithms::prefix_sum(helperList.write().buffer(), prefixHelper.write().buffer(), mHiddenData->mData.length(), helperListLength);
	auto editList = gpu_list<4ui64>().request_length(mHiddenData->mData.requested_length());
	shader_provider::find_value_changes(helperList.buffer(), editList.write().buffer(), mHiddenData->mData.length(), editList.write().length());
	mHiddenData->apply_edit(editList, this);
}

template<class DataList>
inline pbd::indexed_list<DataList> pbd::indexed_list<DataList>::duplicate_these()
{
	auto editList   = gpu_list<4ui64>().request_length(mHiddenData->mData.requested_length()).set_length(mHiddenData->mData.length());
	auto newIndices = gpu_list<4ui64>().request_length(mIndexList.requested_length());
	shader_provider::write_sequence(editList.write().buffer(), mHiddenData->mData.length(), 0, 1);
	editList += mIndexList;
	shader_provider::write_increasing_sequence_from_to(newIndices.write().buffer(), newIndices.write().length(), mHiddenData->mData.length(), editList.length(), mIndexList.length());
	mHiddenData->apply_edit(editList, this);
	auto result = *this;
	result.mIndexList = newIndices;
	return result;
}

template<class DataList>
inline bool pbd::indexed_list<DataList>::empty() const
{
	return mIndexList.empty();
}

template<class DataList>
inline avk::buffer& pbd::indexed_list<DataList>::length() const
{
	return mIndexList.length();
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::set_length(size_t aLength)
{
	mIndexList.set_length(aLength);
	return *this;
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::set_length(const avk::buffer& aLength)
{
	mIndexList.set_length(aLength);
	return *this;
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::request_length(size_t aLength)
{
	mIndexList.request_length(aLength);
	return *this;
}

template<class DataList>
inline size_t pbd::indexed_list<DataList>::requested_length()
{
	return mIndexList.requested_length();
}

template<class DataList>
inline pbd::indexed_list<DataList> pbd::indexed_list<DataList>::increase_length(size_t aAddedLength)
{
	auto result = indexed_list().share_hidden_data_from(*this).request_length(mIndexList.requested_length());
	if (aAddedLength == 0) return result.write();
	mHiddenData->mData.write().set_length(shader_provider::write_increasing_sequence(result.write().index_buffer(), result.write().length(), mHiddenData->mData.write().length(), static_cast<uint32_t>(mHiddenData->mData.requested_length()), static_cast<uint32_t>(aAddedLength)));
	result.mSorted = true;
	*this += result;
	return result;
}

template<class DataList>
inline void pbd::indexed_list<DataList>::apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource)
{
	mSorted = false;
	if (&mIndexList != aEditSource)                     mIndexList.apply_edit(aEditList, this);
	if (mOwner      != aEditSource && mOwner != nullptr)   mOwner->apply_edit(aEditList, this);
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::operator=(const indexed_list& aRhs)
{
	mHiddenData->remove_owner(this);
	mIndexList = aRhs.mIndexList;
	mHiddenData = aRhs.mHiddenData;
	mIndexList.set_owner(this);
	mHiddenData->add_owner(this);
	mSorted = aRhs.mSorted;
	return *this;
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::operator+=(const indexed_list& aRhs)
{
	if (mHiddenData->mData.empty()) {
		share_hidden_data_from(aRhs);
	}
	mIndexList += aRhs.mIndexList;
	mSorted = false;
	return *this;
}

template<class DataList>
inline pbd::indexed_list<DataList> pbd::indexed_list<DataList>::operator+(const indexed_list& aRhs) const
{
	auto result = *this;
	result += aRhs;
	return result;
}

template<class DataList>
inline void pbd::indexed_list<DataList>::set_owner(list_interface<gpu_list<4ui64>>* aOwner)
{
	mOwner = aOwner;
}

template<class DataList>
inline avk::buffer& pbd::indexed_list<DataList>::index_buffer() const
{
	return mIndexList.buffer();
}

template<class DataList>
inline DataList& pbd::indexed_list<DataList>::hidden_list()
{
	return mHiddenData->mData;
}

template<class DataList>
inline std::vector<uint32_t> pbd::indexed_list<DataList>::index_read(bool aBeyondLength) const
{
	return mIndexList.read<uint32_t>(aBeyondLength);
}

template<class DataList>
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::write()
{
	mIndexList.write();
	mSorted = false;
	return *this;
}

template<class DataList>
inline void pbd::indexed_list<DataList>::sort(size_t aValueUpperBound)
{
	if (!mSorted) mIndexList.sort(aValueUpperBound);
	mSorted = true;
}

template<class DataList>
inline void pbd::indexed_list<DataList>::sort()
{
	sort(hidden_list().requested_length());
}

template<class DataList>
inline void pbd::indexed_list<DataList>::apply_hidden_edit(gpu_list<4ui64>& aEditList)
{
	if (empty()) return;

	auto newEditList         = gpu_list<4>().request_length(mIndexList.requested_length());
	auto indexListEqualities = gpu_list<4>().request_length(mIndexList.requested_length());
	auto hiddenIdToIdxListId = gpu_list<4>().request_length(mHiddenData->mData.requested_length());

	// optimization in atomic_swap: no initialization necessary
//	shader_provider::write_sequence(indexListEqualities.write().buffer(), mIndexList.length(), 0, 1);
	shader_provider::write_sequence(hiddenIdToIdxListId.write().buffer(), mHiddenData->mData.length(), std::numeric_limits<uint32_t>().max(), 0);

	shader_provider::atomic_swap(mIndexList.buffer(), indexListEqualities.write().buffer(), hiddenIdToIdxListId.write().buffer(), mIndexList.length());
	mIndexList.set_length(0);
	shader_provider::generate_new_index_and_edit_list(aEditList.buffer(), hiddenIdToIdxListId.buffer(), indexListEqualities.buffer(), write().index_buffer(), newEditList.write().buffer(), aEditList.length(), write().mIndexList.length(), static_cast<uint32_t>(requested_length()));
	if (mOwner == nullptr) return;
	
	newEditList.set_length(mIndexList.length());
	mOwner->apply_edit(newEditList, this);
}
