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
		const shader_provider::changing_length changing_length();
		indexed_list& set_length(size_t aLength);
		indexed_list& request_length(size_t aLength);
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
	auto helper_list_length = mHiddenData->mData.buffer()->meta_at_index<avk::buffer_meta>().total_size();
	auto helperList   = gpu_list<4ui64>();
	auto prefix_helper = gpu_list<4ui64>();
	helperList.request_length(helper_list_length);
	prefix_helper.request_length(list_manipulation::prefixSum_calculateNeededHelperListLength(helper_list_length));
	list_manipulation::fillList_singleValue(helperList.write().buffer(), mHiddenData->mData.length(), 1u);
	list_manipulation::scatteredWrite_value(mIndexList.getReadBuffer(), 0u, mIndexList.length(), helperList.getWriteBuffer());
	resize(0);
	list_manipulation::prefixSum(helperList.getWriteBuffer(), prefix_helper.getWriteBuffer(), helperList.length());
	auto editList = gpu_list<4ui64>();
	list_manipulation::scatteredWriteAfterPrefixSum_indices(helperList.getReadBuffer(), helperList.length(), editList.getWriteBuffer(mHiddenData->mData.length()));
	mHiddenData->mData.apply_edit(editList, this);
}

template<class DataList>
inline avk::buffer& pbd::indexed_list<DataList>::length() const
{
	return mIndexList.length();
}

template<class DataList>
inline const shader_provider::changing_length pbd::indexed_list<DataList>::changing_length()
{
	return mIndexList.changing_length();
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
inline pbd::indexed_list<DataList> pbd::indexed_list<DataList>::increase_length(size_t aAddedLength)
{
	auto result = indexed_list().share_hidden_data_from(*this).request_length(mIndexList.requested_length());
	shader_provider::write_increasing_sequence(result.write().index_buffer(), result.write().length(), mHiddenData->mData.changing_length(), mHiddenData->mData.requested_length(), aAddedLength);
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
}

template<class DataList>
inline void pbd::indexed_list<DataList>::apply_hidden_edit(gpu_list<4ui64>& aEditList)
{
	auto sortMapping = gpu_list<4ui64>();

	if (!mSorted)
	{
		auto oldIndexList = mIndexList;
		auto sortHelper = gpu_list<4ui64>().set_length(algorithms::sort_calculateNeededHelperListLength(mIndexList.length()));
		auto increasing = gpu_list<4ui64>();
		sortHelper.resize(list_manipulation::sort_calculateNeededHelperListLength(mIndexList.length()));
		increasing.resize(mIndexList.length());
		sortMapping.resize(mIndexList.length());
		list_manipulation::writeIncreasingSequence(increasing.getWriteBuffer(), 0, 0, increasing.length());
		list_manipulation::sort(oldIndexList.getWriteBuffer(), increasing.getWriteBuffer(), sortHelper.getWriteBuffer(), mIndexList.length(), mIndexList.getWriteBuffer(), sortMapping.getWriteBuffer());
	}

	// generate histogram bin start list and histogram bin end list from index list; example:
	// index list: 0, 2, 3, 3    =>    histogram bin start list: 0, 0, 1, 2    histogram bin end list: 1, 0, 2, 4

	auto indexListContentUpperBound = 32768u; // TODO length of hidden list before edit

	auto binStartIdx = gpu_list<4ui64>();
	auto binEndIdx   = gpu_list<4ui64>();
	binStartIdx.resize(indexListContentUpperBound);
	binEndIdx.resize(indexListContentUpperBound);

	list_manipulation::generateBinLists(mIndexList.getReadBuffer(), mIndexList.length(), binStartIdx.getWriteBuffer(), binEndIdx.getWriteBuffer());

	// generate target index list from edit list and the two histogram bin lists; example:
	// edit list: 0, 3, 1, 2, 4    &    histogram bin start list: 0, 0, 1, 2    &    histogram bin end list: 1, 0, 2, 4    =>    target index list: 1, 3, 3, 4, 4
	//
	// 1. for every i: targetIndexList[i] = binEndIdx[editList[i]] - binStartIdx[editList[i]]
	// 2. prefix sum over targetIndexList

	auto targetIndexList = gpu_list<4ui64>();
	targetIndexList.resize(editList.length());
	list_manipulation::computeBinSizes(binStartIdx.getReadBuffer(), binEndIdx.getReadBuffer(), editList.getReadBuffer(), editList.length(), targetIndexList.getWriteBuffer());

	auto& prefixHelper = binEndIdx; // re-use list
	prefixHelper.resize(list_manipulation::prefixSum_calculateNeededHelperListLength(targetIndexList.length()));
	list_manipulation::prefixSum(targetIndexList.getWriteBuffer(), prefixHelper.getWriteBuffer(), targetIndexList.length());

	// generate new index list; example:
	// target index list: 1, 3, 3, 4, 4    =>    new index list: 0, 1, 1, 3
	//
	// 1. for every i: for every j in [targetIndexList[i - 1]; targetIndexList[i]): newIndexList[j] = i

	auto howShouldIKnow = 32768u; // TODO upper bound to new index list length - alternatively read last value from targetIndexList to get exact length
	mIndexList.resize(0); // binToValueList() will determine the new length
	list_manipulation::binToValueList(targetIndexList.getReadBuffer(), targetIndexList.length(), mIndexList.getWriteBuffer(howShouldIKnow));

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
	newEditList.resize(mIndexList.length());
	list_manipulation::generateNewEditList(binStartIdx.getReadBuffer(), editList.getReadBuffer(), mIndexList.getReadBuffer(), targetIndexList.getReadBuffer(), mIndexList.length(), newEditList.getWriteBuffer());

	if (!mSorted)
	{
		mSorted = true;
		sortMapping.apply_edit(newEditList, nullptr);
		mOwner->apply_edit(sortMapping, this);
	}
	else
	{
		mOwner->apply_edit(newEditList, this);
	}
}
