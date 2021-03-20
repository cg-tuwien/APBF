#pragma once

namespace pbd
{
	/// <summary><para>Consists of a buffer containing a list of data, and a buffer containing the length of the list.</para>
	/// <para>Use the static function <seealso cref="gpu_list_data::get_list"/> to get a gpu_list_data wrapped in a shared pointer.
	/// When there are no other references left, the gpu_list_data will be recycled or destroyed.</para></summary>
	class gpu_list_data
	{
	public:
		gpu_list_data(avk::buffer aBuffer, avk::buffer aLength) : mBuffer{ std::move(aBuffer) }, mLength{ std::move(aLength) } {}
		avk::buffer mBuffer;
		avk::buffer mLength;

		// static buffer cache
		static std::shared_ptr<gpu_list_data> get_list(size_t aMinLength, uint32_t aStride);
		static void garbage_collection();
		static void cleanup();

	private:
		size_t size();

		struct list_entry
		{
			std::shared_ptr<gpu_list_data> mGpuListData;
			uint32_t mTimeUnused; // how many garbage collections ago was it used last?
		};

		static std::list<list_entry> mReservedLists;
		static uint32_t mGarbageCollectionCountBeforeDeletion; // if mTimeUnused reaches this value, delete this buffer
	};
}
