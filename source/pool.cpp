#include "pool.h"
#include "initialize.h"
#include "measurements.h"
#include "settings.h"

pool::pool(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius) :
	mParticles(100000),
	mTransfers(100),
	mTimeMachine(mParticles, mParticles.hidden_list(), mFluid, mTransfers.hidden_list(), mDeltaTime)
{
	shader_provider::start_recording();
	mParticles.request_length(100000);
	mFluid.request_length(100000);
//	mTransfers.request_length(100); // not even necessary because index list never gets used
	mNeighborsFluid.request_length(10000000);
	mTransfers.hidden_list().get<pbd::hidden_transfers::id::source>().share_hidden_data_from(mParticles);
	mTransfers.hidden_list().get<pbd::hidden_transfers::id::target>().share_hidden_data_from(mParticles);
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_box_shape(mParticles, aMin + glm::vec3(2, 2, 2), aMax - glm::vec3(2, 4, 2), aRadius);
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.length(), 4, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::boundary_distance>().write().buffer(), mFluid.length(), 0, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::target_radius>().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence(mFluid.get<pbd::fluid::id::transferring>().write().buffer(), mFluid.length(), 0, 0);
	mVelocityHandling.add_particles(mParticles, glm::vec3(0, -10, 0)); // TODO: let mVelocityHandling and mBoxCollision only store pointers to mParticles? Especially now with time_machine.
	mBoxCollision.add_particles(mParticles);
	mBoxCollision.add_box(aMin, glm::vec3(aMin.x + 2, aMax.y, aMax.z));
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z));
	mBoxCollision.add_box(aMin, glm::vec3(aMax.x, aMax.y, aMin.z + 2));
	mBoxCollision.add_box(glm::vec3(aMax.x - 2, aMin.y, aMin.z), aMax);
	mBoxCollision.add_box(glm::vec3(aMin.x, aMax.y - 2, aMin.z), aMax);
	mBoxCollision.add_box(glm::vec3(aMin.x, aMin.y, aMax.z - 2), aMax);
	mSpreadKernelWidth.set_data(&mFluid, &mNeighborsFluid);
	mIncompressibility.set_data(&mFluid, &mNeighborsFluid);
	mUpdateTransfers.set_data(&mFluid, &mNeighborsFluid, &mTransfers);
	mParticleTransfer.set_data(&mFluid, &mTransfers);
	mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::kernel_width>(), &mNeighborsFluid);
#if NEIGHBORHOOD_TYPE == 1
	mNeighborhoodFluid.set_position_range(aMin, aMax, 6u);
#endif
	mNeighborhoodFluid.set_range_scale(1.5f);
	mTimeMachine.set_max_keyframes(8);
	mTimeMachine.set_keyframe_interval(120);
	mTimeMachine.save_state();
	shader_provider::end_recording();
}

void pool::update(float aDeltaTime)
{
	mDeltaTime = aDeltaTime;
	if (!mTimeMachine.step_forward()) {
		mVelocityHandling.apply(mDeltaTime);
		if (pbd::settings::merge || pbd::settings::split) {
			mParticleTransfer.apply(mDeltaTime);
		}
		measurements::record_timing_interval_start("Neighborhood");
		mNeighborhoodFluid.apply();
		measurements::record_timing_interval_end("Neighborhood");
		mSpreadKernelWidth.apply();

		for (uint32_t i = 0u; i < pbd::settings::solverIterations; i++) {
			mBoxCollision.apply();
			mIncompressibility.apply();
		}

		if (pbd::settings::merge || pbd::settings::split || pbd::settings::baseKernelWidthOnTargetRadius || pbd::settings::color == 1 || pbd::settings::color == 2) {
			mUpdateTransfers.apply();
		}

		mTimeMachine.save_state();
	}
}

pbd::particles& pool::particles()
{
	return mParticles;
}

pbd::fluid& pool::fluid()
{
	return mFluid;
}

pbd::neighbors& pool::neighbors()
{
	return mNeighborsFluid;
}
