#pragma once

#include "indexed_list.h"
#include "uninterleaved_list.h"

namespace pbd {
	template<class... Data>
	class time_machine {
	public:
		// if we are in the past, load data from next step; if we stepped onto an existing keyframe, also load gpu data and return true
		auto step_forward() { auto past = in_past(); ++mCurrentStep; mPresentStep += past ? 0u : 1u; return past && on_keyframe(); }
		// if we are in the present, save the current state
		void save_state() {}
		// attempt to jump back into the past by one keyframe
		auto jump_back() { if (on_keyframe() && mCurrentStep + mKeyframeInterval * (mMaxKeyframes - 1u) <= mPresentStep || mCurrentStep == 0u) return false; mCurrentStep = (mCurrentStep - 1u) / mKeyframeInterval * mKeyframeInterval; return true; }

		auto max_keyframes() { return mMaxKeyframes; }
		void set_max_keyframes(uint32_t aMaxKeyframes) { mMaxKeyframes = aMaxKeyframes; }
		auto keyframe_interval() { return mKeyframeInterval; }
		void set_keyframe_interval(uint32_t aKeyframeInterval) { mKeyframeInterval = aKeyframeInterval; }
		auto on_keyframe() { return mCurrentStep % mKeyframeInterval == 0u; }
		auto history_index() { return std::min(mCurrentStep, mCurrentStep - oldest_history_keyframe_step()); }
		auto sparse_history_index() { return history_index() / mKeyframeInterval; }
		auto in_past() { return mCurrentStep != mPresentStep; }
		auto keyframe_step_for(uint32_t aStep) { return aStep / mKeyframeInterval * mKeyframeInterval; }
		// wrong result if no full history yet - let the min() in history_index handle this case!
		auto oldest_history_keyframe_step() { return keyframe_step_for(mPresentStep) - (mMaxKeyframes - 1u) * mKeyframeInterval; }

	private:
		uint32_t mMaxKeyframes = 5u;
		uint32_t mKeyframeInterval = 120u;
		uint32_t mCurrentStep = 0u;
		uint32_t mPresentStep = 0u;
	};





	template<class FirstData, class... Data>
	class time_machine<FirstData, Data...>
	{
	public:
		time_machine(FirstData& aFirstData, Data&... aData);
		auto step_forward();
		void save_state();
		auto jump_back();

		auto max_keyframes() { return mRest.max_keyframes(); }
		void set_max_keyframes(uint32_t aMaxKeyframes) { mRest.set_max_keyframes(aMaxKeyframes); }
		auto keyframe_interval() { return mRest.keyframe_interval(); }
		void set_keyframe_interval(uint32_t aKeyframeInterval) { mRest.set_keyframe_interval(aKeyframeInterval); mDataHistory.clear(); }
		auto on_keyframe() { return mRest.on_keyframe(); }
		auto history_index() { return mRest.history_index(); }
		auto sparse_history_index() { return mRest.sparse_history_index(); }
		auto in_past() { return mRest.in_past(); }

	private:
		void load_state();

		bool mSparse;
		FirstData& mDataPointer;
		std::deque<FirstData> mDataHistory;
		time_machine<Data...> mRest;
	};



	template<class FirstData, class... Data>
	inline pbd::time_machine<FirstData, Data...>::time_machine(FirstData& aFirstData, Data&... aData) :
		mDataPointer(aFirstData),
		mRest(aData...),
		mSparse(is_gpu_list<FirstData>::value)
	{}

	template<class FirstData, class... Data>
	inline auto pbd::time_machine<FirstData, Data...>::step_forward()
	{
		auto past = mRest.in_past();
		auto result = mRest.step_forward();
		if (past && (!mSparse || result)) {
			load_state();
		}
		else if (!past && mRest.on_keyframe()) {
			auto newHistoryLength = (mRest.max_keyframes() - 1u) * (mSparse ? 1u : mRest.keyframe_interval());
			auto popCount = newHistoryLength > mDataHistory.size() ? 0u : (mDataHistory.size() - newHistoryLength);
			for (auto i = popCount; i > 0u; i--) mDataHistory.pop_front();
		}
		return result;
	}

	template<class FirstData, class... Data>
	inline void pbd::time_machine<FirstData, Data...>::save_state()
	{
		auto index = mSparse ? mRest.sparse_history_index() : mRest.history_index();
		if (mDataHistory.size() <= index) {
			mDataHistory.push_back(mDataPointer);
		}
		mRest.save_state();
	}

	template<class FirstData, class... Data>
	inline auto pbd::time_machine<FirstData, Data...>::jump_back()
	{
		if (mRest.jump_back()) { load_state(); return true; }
		return false;
	}

	template<class FirstData, class... Data>
	inline void pbd::time_machine<FirstData, Data...>::load_state()
	{
		mDataPointer = mDataHistory[mSparse ? mRest.sparse_history_index() : mRest.history_index()];
	}

	template<class FirstData, class... Data> time_machine(FirstData& aFirstData, Data&... aData) -> time_machine<FirstData, Data...>;


	template<class T>
	struct is_gpu_list : std::false_type {};
	template<size_t T>
	struct is_gpu_list<pbd::gpu_list<T>> : std::true_type {};
	template<class T>
	struct is_gpu_list<pbd::indexed_list<T>> : std::true_type {};
	template<class... T>
	struct is_gpu_list<pbd::uninterleaved_list<T...>> : std::true_type {};
}
