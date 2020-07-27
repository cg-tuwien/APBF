#pragma once

#include <gvk.hpp>

class shader_provider
{
public:
	// TODO maybe add an init_all() function
	static void set_queue(avk::queue& aQueue);
	static void start_recording();
	static void end_recording();
	static avk::command_buffer& cmd_bfr();
	static void roundandround(const avk::buffer& aAppData, const avk::buffer& aParticles, const avk::buffer& aAabbs, uint32_t aParticleCount);
	static void append_list(const avk::buffer& aTargetList, const avk::buffer& aAppendingList, const avk::buffer& aTargetListLength, const avk::buffer& aAppendingListLength);

	static void sync_after_compute();
	static void sync_after_transfer();
private:
	static avk::descriptor_cache& descriptor_cache();
	static const avk::buffer& workgroup_count_buffer();
	static void dispatch(uint32_t aX, uint32_t aY, uint32_t aZ, uint32_t aLocalSizeX = 256, uint32_t aLocalSizeY = 1, uint32_t aLocalSizeZ = 1);
	static void prepare_dispatch_indirect(const avk::buffer& aXyz, uint32_t aLocalSizeX = 256, uint32_t aLocalSizeY = 1, uint32_t aLocalSizeZ = 1);
	static void dispatch_indirect();

	static avk::command_buffer mCmdBfr;
	static avk::queue* mQueue;
};
