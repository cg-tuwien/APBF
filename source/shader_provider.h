#pragma once

#include <exekutor.hpp>

class shader_provider
{
public:
	// TODO maybe add an init_all() function
	static void set_queue(ak::queue& aQueue);
	static void start_recording();
	static void end_recording();
	static ak::command_buffer& cmd_bfr();
	static void roundandround(const ak::buffer& aAppData, const ak::buffer& aParticles, const ak::buffer& aAabbs, uint32_t aParticleCount);
	static void append_list(const ak::buffer& aTargetList, const ak::buffer& aAppendingList, const ak::buffer& aTargetListLength, const ak::buffer& aAppendingListLength);

	static void sync_after_compute();
	static void sync_after_transfer();
private:
	static ak::descriptor_cache& descriptor_cache();
	static const ak::buffer& workgroup_count_buffer();
	static void dispatch(uint32_t aX, uint32_t aY, uint32_t aZ, uint32_t aLocalSizeX = 256, uint32_t aLocalSizeY = 1, uint32_t aLocalSizeZ = 1);
	static void prepare_dispatch_indirect(const ak::buffer& aXyz, uint32_t aLocalSizeX = 256, uint32_t aLocalSizeY = 1, uint32_t aLocalSizeZ = 1);
	static void dispatch_indirect();

	static ak::command_buffer mCmdBfr;
	static ak::queue* mQueue;
};
