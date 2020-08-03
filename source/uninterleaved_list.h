#pragma once

#include "gpu_list.h"

namespace pbd
{
	template<class NameEnum, class... Lists>
	class uninterleaved_list :
		public list_interface<gpu_list<4ui64>>
	{
	public:
		using id = NameEnum;

		uninterleaved_list();
		uninterleaved_list(const uninterleaved_list& aUninterleavedList);
		~uninterleaved_list() = default;

		bool empty() const;
		avk::buffer& length() const;
		uninterleaved_list& set_length(size_t aLength);
		uninterleaved_list& set_length(const avk::buffer& aLength);
		uninterleaved_list& request_length(size_t aLength);
		size_t requested_length();
		void apply_edit(gpu_list<4ui64> & aEditList, list_interface<gpu_list<4ui64>>* aEditSource) override;

		uninterleaved_list& operator=(const uninterleaved_list& aRhs);
		uninterleaved_list& operator+=(const uninterleaved_list& aRhs);
		uninterleaved_list operator+(const uninterleaved_list& aRhs) const;

		uninterleaved_list& write();

		uninterleaved_list& set_owner(list_interface<gpu_list<4ui64>>* aOwner);
		template<NameEnum E>
		constexpr auto& get()
		{
			return std::get<static_cast<int>(E)>(mLists);
		}
		template<NameEnum E>
		constexpr auto& get() const
		{
			return std::get<static_cast<int>(E)>(mLists);
		}

	private:
		template<size_t... Is>
		void set_owner(std::index_sequence<Is...>) {
			(std::get<Is>(mLists).set_owner(this), ...); // C++17
			//int arr[] = { (std::get<Is>(mLists).setOwner(this), 0)... };
		}
		template<size_t... Is>
		void set_length(size_t aLength, std::index_sequence<Is...>) {
			(std::get<Is>(mLists).set_length(aLength), ...);
			//int arr[] = { (std::get<Is>(mLists).set_length(aLength), 0)... };
		}
		template<size_t... Is>
		void set_length(const avk::buffer& aLength, std::index_sequence<Is...>) {
			(std::get<Is>(mLists).set_length(aLength), ...);
			//int arr[] = { (std::get<Is>(mLists).set_length(aLength), 0)... };
		}
		template<size_t... Is>
		void request_length(size_t aLength, std::index_sequence<Is...>) {
			(std::get<Is>(mLists).request_length(aLength), ...);
			//int arr[] = { (std::get<Is>(mLists).request_length(aLength), 0)... };
		}
		template<size_t... Is>
		void apply_edit(gpu_list<4ui64>& aEditList, list_interface<gpu_list<4ui64>>* aEditSource, std::index_sequence<Is...>) {
			((&std::get<Is>(mLists) == aEditSource ? 0 : (std::get<Is>(mLists).apply_edit(aEditList, this), 0)), ...);
			//int arr[] = { (&std::get<Is>(mLists) == aEditSource ? 0 : (std::get<Is>(mLists).apply_edit(aEditList, this), 0))... };
		}
		template<size_t... Is>
		void add(const uninterleaved_list& aRhs, std::index_sequence<Is...>) {
			(std::get<Is>(mLists).operator+=(std::get<Is>(aRhs.mLists)), ...);
			//int arr[] = { (std::get<Is>(mLists).operator+=(std::get<Is>(aRhs.mLists)), 0)... };
		}
		template<size_t... Is>
		void write(std::index_sequence<Is...>) {
			(std::get<Is>(mLists).write(), ...);
			//int arr[] = { (std::get<Is>(mLists).write(), 0)... };
		}

		std::tuple<Lists...> mLists;
		list_interface<gpu_list<4ui64>>* mOwner = nullptr;
	};
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>::uninterleaved_list()
{
	set_owner(std::make_index_sequence<sizeof...(Lists)>());
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>::uninterleaved_list(const uninterleaved_list& aUninterleavedList) :
	mLists(aUninterleavedList.mLists)
{
	set_owner(std::make_index_sequence<sizeof...(Lists)>());
}

template<class NameEnum, class ...Lists>
inline bool pbd::uninterleaved_list<NameEnum, Lists...>::empty() const
{
	return std::get<0>(mLists).empty();
}

template<class NameEnum, class... Lists>
inline avk::buffer& pbd::uninterleaved_list<NameEnum, Lists...>::length() const
{
	return std::get<0>(mLists).length();
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::set_length(size_t aLength)
{
	set_length(aLength, std::make_index_sequence<sizeof...(Lists)>());
	return *this;
}

template<class NameEnum, class ...Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::set_length(const avk::buffer& aLength)
{
	set_length(aLength, std::make_index_sequence<sizeof...(Lists)>());
	return *this;
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::request_length(size_t aLength)
{
	request_length(aLength, std::make_index_sequence<sizeof...(Lists)>());
	return *this;
}

template<class NameEnum, class ...Lists>
inline size_t pbd::uninterleaved_list<NameEnum, Lists...>::requested_length()
{
	// TODO maybe return minimum of all requested lengths
	return std::get<0>(mLists).requested_length();
}

template<class NameEnum, class... Lists>
inline void pbd::uninterleaved_list<NameEnum, Lists...>::apply_edit(pbd::gpu_list<4ui64>& aEditList, list_interface<pbd::gpu_list<4ui64>>* aEditSource)
{
	if (mOwner != aEditSource && mOwner != nullptr) mOwner->apply_edit(aEditList, this);
	apply_edit(aEditList, aEditSource, std::make_index_sequence<sizeof...(Lists)>());
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::operator=(const uninterleaved_list& aRhs)
{
	mLists = aRhs.mLists;
	set_owner(std::make_index_sequence<sizeof...(Lists)>());
	return *this;
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::operator+=(const uninterleaved_list& aRhs)
{
	add(aRhs, std::make_index_sequence<sizeof...(Lists)>());
	return *this;
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...> pbd::uninterleaved_list<NameEnum, Lists...>::operator+(const uninterleaved_list& aRhs) const
{
	auto result = *this;
	result += aRhs;
	return result;
}

template<class NameEnum, class ...Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::write()
{
	write(std::make_index_sequence<sizeof...(Lists)>());
	return *this;
}

template<class NameEnum, class... Lists>
inline pbd::uninterleaved_list<NameEnum, Lists...>& pbd::uninterleaved_list<NameEnum, Lists...>::set_owner(list_interface<pbd::gpu_list<4ui64>>* aOwner)
{
	mOwner = aOwner;
	return *this;
}
