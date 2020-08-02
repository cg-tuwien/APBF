#pragma once

#include "gpu_list.h"

namespace pbd
{
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

		avk::buffer& length() const;
		indexed_list& set_length(size_t aLength);
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

		/// <summary><para>Request write access to the list. The intended use is: indexed_list.write().index_buffer() and indexed_list.write().length().</para></summary>
		indexed_list& write();

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
				if (&mData != aEditSource) mData.apply_edit(aEditList, this);
				for (auto aOwner : mOwners) if (aOwner != aEditSource) aOwner->apply_hidden_edit(aEditList);
			}

			DataList mData;
			std::list<indexed_list*> mOwners;
		};

		void apply_hidden_edit(gpu_list<4ui64>& aEditList);

		gpu_list<4ui64> mIndexList;
		std::shared_ptr<hidden_data> mHiddenData;
		list_interface<gpu_list<4ui64>>* mOwner = nullptr;
		bool mSorted;
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
	auto helperListLength = mHiddenData->mData.buffer()->meta_at_index<avk::buffer_meta>().total_size();
	auto helperList   = gpu_list<4ui64>().request_length(helperListLength);
	auto prefixHelper = gpu_list<4ui64>().request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(helperListLength));
	shader_provider::write_sequence(helperList.write().buffer(), mHiddenData->mData.length(), 1u, 0u);
	shader_provider::scattered_write(mIndexList.buffer(), helperList.write().buffer(), mIndexList.length(), 0u);
	mIndexList.set_length(0);
	algorithms::prefix_sum(helperList.write().buffer(), prefixHelper.write().buffer(), mHiddenData->mData.length());
	auto editList = gpu_list<4ui64>().request_length(mHiddenData->mData.requested_length());
	shader_provider::find_value_changes(helperList.buffer(), editList.write().buffer(), mHiddenData->mData.length(), editList.write().length());
	mHiddenData->mData.apply_edit(editList, this);
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
	mHiddenData->mData.write().set_length(shader_provider::write_increasing_sequence(result.write().index_buffer(), result.write().length(), mHiddenData->mData.write().length(), mHiddenData->mData.requested_length(), aAddedLength));
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
inline pbd::indexed_list<DataList>& pbd::indexed_list<DataList>::write()
{
	mIndexList.write();
	mSorted = false;
	return *this;
}

template<class DataList>
inline void pbd::indexed_list<DataList>::apply_hidden_edit(gpu_list<4ui64>& aEditList)
{
	auto sortMapping = gpu_list<4ui64>();
	auto indexListContentUpperBound = mHiddenData->mData.requested_length();

	if (!mSorted)
	{
		auto oldIndexList = mIndexList;
		auto sortHelper = gpu_list<4ui64>().request_length(algorithms::sort_calculate_needed_helper_list_length(mIndexList.requested_length()));
		auto increasing = gpu_list<4ui64>().request_length(mIndexList.requested_length());
		sortMapping.request_length(mIndexList.requested_length());
		shader_provider::write_sequence(increasing.write().buffer(), mIndexList.length(), 0u, 1u);
		algorithms::sort(oldIndexList.write().buffer(), increasing.write().buffer(), sortHelper.write().buffer(), mIndexList.write().length(), mIndexList.write().buffer(), sortMapping.write().buffer(), indexListContentUpperBound);
	}

	// generate histogram bin start list and histogram bin end list from index list; example:
	// index list: 0, 2, 3, 3    =>    histogram bin start list: 0, 0, 1, 2    histogram bin end list: 1, 0, 2, 4


	auto binStartIdx = gpu_list<4ui64>().set_length(indexListContentUpperBound);
	auto binEndIdx   = gpu_list<4ui64>().set_length(indexListContentUpperBound);

	shader_provider::write_sequence(binStartIdx.write().buffer(), binStartIdx.write().length(), 0u, 0u);
	shader_provider::write_sequence(  binEndIdx.write().buffer(),   binEndIdx.write().length(), 0u, 0u);
	shader_provider::find_value_ranges(mIndexList.buffer(), binStartIdx.write().buffer(), binEndIdx.write().buffer(), mIndexList.length());

	// generate target index list from edit list and the two histogram bin lists; example:
	// edit list: 0, 3, 1, 2, 4    &    histogram bin start list: 0, 0, 1, 2    &    histogram bin end list: 1, 0, 2, 4    =>    target index list: 1, 3, 3, 4, 4
	//
	// 1. for every i: targetIndexList[i] = binEndIdx[editList[i]] - binStartIdx[editList[i]]
	// 2. prefix sum over targetIndexList

	auto targetIndexList = gpu_list<4ui64>().request_length(aEditList.requested_length());
	shader_provider::indexed_subtract(aEditList.buffer(), binEndIdx.buffer(), binStartIdx.buffer(), targetIndexList.write().buffer(), aEditList.length());

	auto& prefixHelper = binEndIdx; // re-use list
	prefixHelper.request_length(algorithms::prefix_sum_calculate_needed_helper_list_length(targetIndexList.requested_length()));
	algorithms::prefix_sum(targetIndexList.write().buffer(), prefixHelper.write().buffer(), aEditList.length());

	// generate new index list; example:
	// target index list: 1, 3, 3, 4, 4    =>    new index list: 0, 1, 1, 3
	//
	// 1. for every i: for every j in [targetIndexList[i - 1]; targetIndexList[i]): newIndexList[j] = i

	set_length(0); // if aEditList has length 0, the following compute shader is not executed and would not correctly set the new length to 0
	shader_provider::generate_new_index_list(targetIndexList.buffer(), mIndexList.write().buffer(), aEditList.length(), mIndexList.write().length());

	if (mOwner == nullptr)
	{
		mSorted = true;
		return;
	}

	// generate new edit list; example:
	// histogram bin start list: 0, x, 1, 2    &    edit list: 0, 3, 1, 2, 4    &    new index list: 0, 1, 1, 3    &    target index list: 1, 3, 3, 4, 4    =>    new edit list: 0, 2, 3, 1
	//
	// 1. for every i: newEditList[i] = binStartIdx[editList[newIndexList[i]]] + i - targetIndexList[newIndexList[i] - 1]

	auto& newEditList = binEndIdx; // re-use list
	newEditList.request_length(mIndexList.requested_length()).set_length(length());
	shader_provider::generate_new_edit_list(mIndexList.buffer(), aEditList.buffer(), binStartIdx.buffer(), targetIndexList.buffer(), newEditList.write().buffer(), mIndexList.length());

	if (!mSorted) {
		mSorted = true;
		sortMapping.apply_edit(newEditList, nullptr);
		mOwner->apply_edit(sortMapping, this);
	} else {
		mOwner->apply_edit(newEditList, this);
	}
}
