#include "waterfall.h"
#include "initialize.h"
#include "measurements.h"
#include "settings.h"

waterfall::waterfall(const glm::vec3& aMin, const glm::vec3& aMax, float aRadius) :
	mParticles(100000),
	mTransfers(MAX_TRANSFERS),
	mTimeMachine(mParticles, mParticles.hidden_list(), mFluid.get<pbd::fluid::id::particle>(), mFluid.get<pbd::fluid::id::boundariness>(), mFluid.get<pbd::fluid::id::boundary_distance>(), mFluid.get<pbd::fluid::id::kernel_width>(), mFluid.get<pbd::fluid::id::target_radius>(), mTransfers.hidden_list().get<pbd::hidden_transfers::id::time_left>(), mTransfers.hidden_list().get<pbd::hidden_transfers::id::source>(), mTransfers.hidden_list().get<pbd::hidden_transfers::id::target>(), mDeltaTime)
{
	shader_provider::start_recording();
	mParticles.request_length(100000);
	mFluid.request_length(100000);
//	mTransfers.request_length(100); // not even necessary because index list never gets used
	mNeighborsFluid.request_length(10000000); // TODO why does a longer neighbor list lead to higher storage requirements for each time machine keyframe? Probably because these long buffers get re-used for other lists which then get stored in keyframes
	mParticles.write();               //
	mParticles.hidden_list().write(); // TODO workaround for the case that no particles are created => find better solution
	mTransfers.hidden_list().get<pbd::hidden_transfers::id::source>().share_hidden_data_from(mParticles);
	mTransfers.hidden_list().get<pbd::hidden_transfers::id::target>().share_hidden_data_from(mParticles);
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_box_shape(mParticles, aMin + glm::vec3(2, 2, 2), aMax - glm::vec3(2, 4, 2), aRadius);
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.length(), aRadius * static_cast<float>(KERNEL_SCALE), 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::target_radius>().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::boundariness>().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence(mFluid.get<pbd::fluid::id::boundary_distance>().write().buffer(), mFluid.length(), static_cast<uint32_t>(aRadius * POS_RESOLUTION), 0);

	mVelocityHandling.set_data(&mParticles);
	mVelocityHandling.set_acceleration(glm::vec3(0, -10, 0));
	mBoxCollision.set_data(&mParticles, &mUcb.box_min(), &mUcb.box_max());
	// top pool
	mUcb.add_box(aMin, glm::vec3(aMin.x + 2, aMax.y, aMax.z));
	mUcb.add_box(aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z));
	mUcb.add_box(glm::vec3(aMax.x - 2, aMin.y, aMin.z), aMax);
	mUcb.add_box(glm::vec3(aMin.x, aMax.y - 2, aMin.z), aMax);
	// bottom pool
	auto shift = aMin - aMax;
	shift.z = 0;
	mUcb.add_box(shift + aMin, glm::vec3(aMin.x + 2, aMax.y, aMax.z) + shift);
	mUcb.add_box(shift + aMin, glm::vec3(aMax.x, aMin.y + 2, aMax.z) + shift);
	mUcb.add_box(shift + glm::vec3(aMax.x - 2, aMin.y, aMin.z), aMax + shift);
	mUcb.add_box(shift + glm::vec3(aMin.x, aMax.y - 2, aMin.z), aMax + shift);
#if DIMENSIONS > 2
	// top pool
	mUcb.add_box(aMin, glm::vec3(aMax.x, aMax.y, aMin.z + 2));
	mUcb.add_box(glm::vec3(aMin.x, aMin.y, aMax.z - 2), aMax);
	// bottom pool
	mUcb.add_box(shift + aMin, glm::vec3(aMax.x, aMax.y, aMin.z + 2) + shift);
	mUcb.add_box(shift + glm::vec3(aMin.x, aMin.y, aMax.z - 2), aMax + shift);
#endif
	mSpreadKernelWidth.set_data(&mFluid, &mNeighborsFluid);
	mIncompressibility.set_data(&mFluid, &mNeighborsFluid);
	mUpdateTransfers.set_data(&mFluid, &mNeighborsFluid, &mTransfers);
	mParticleTransfer.set_data(&mFluid, &mTransfers);
	mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::kernel_width>(), &mNeighborsFluid);
#if NEIGHBORHOOD_TYPE == 1
	mNeighborhoodFluid.set_position_range(aMin, aMax, 4u);
#endif
	mNeighborhoodFluid.set_range_scale(pbd::settings::baseKernelWidthOnBoundaryDistance ? 1.0f : 1.5f);
	mTimeMachine.set_max_keyframes(4).set_keyframe_interval(120).enable();
	shader_provider::end_recording();
	mRenderBoxes = true;
}

void waterfall::update(float aDeltaTime)
{
	mDeltaTime = FIXED_TIME_STEP == 0 ? aDeltaTime : FIXED_TIME_STEP;
	if (!mTimeMachine.step_forward()) {
		mVelocityHandling.apply(mDeltaTime);

		if (pbd::settings::merge || pbd::settings::split) {
			mParticleTransfer.apply(mDeltaTime);
		}

		if (pbd::settings::baseKernelWidthOnBoundaryDistance) {
			shader_provider::uint_to_float_with_indexed_lower_bound(mFluid.get<pbd::fluid::id::boundary_distance>().buffer(), mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.get<pbd::fluid::id::particle>().index_buffer(), mFluid.get<pbd::fluid::id::particle>().hidden_list().get<pbd::hidden_particles::id::radius>().buffer(), mFluid.length(), pbd::settings::targetRadiusScaleFactor / static_cast<float>(POS_RESOLUTION), static_cast<float>(KERNEL_SCALE), pbd::settings::kernelWidthAdaptionSpeed);
			//shader_provider::uint_to_float_but_gradual(mFluid.get<pbd::fluid::id::boundary_distance>().buffer(), mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.length(), pbd::settings::targetRadiusScaleFactor / POS_RESOLUTION, pbd::settings::kernelWidthAdaptionSpeed, pbd::settings::smallestTargetRadius * KERNEL_SCALE);
		}

		measurements::record_timing_interval_start("Neighborhood");
		mNeighborhoodFluid.apply();
		measurements::record_timing_interval_end("Neighborhood");

		if (!pbd::settings::baseKernelWidthOnBoundaryDistance) {
			mSpreadKernelWidth.apply();
		}

		for (auto i = 0; i < pbd::settings::solverIterations; i++) {
			mBoxCollision.apply();
			mIncompressibility.apply();
		}

		//if (pbd::settings::merge || pbd::settings::split || pbd::settings::baseKernelWidthOnTargetRadius || pbd::settings::color == 1 || pbd::settings::color == 2) // TODO reactivate?
		{
			mUpdateTransfers.apply();
		}

		mTimeMachine.save_state();
	}
}

pbd::particles& waterfall::particles()
{
	return mParticles;
}

pbd::fluid& waterfall::fluid()
{
	return mFluid;
}

pbd::neighbors& waterfall::neighbors()
{
	return mNeighborsFluid;
}

void waterfall::handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos)
{
	if (mRenderBoxes) mUcb.handle_input(aInverseViewProjection, aCameraPos);
}

void waterfall::render(const glm::mat4& aViewProjection)
{
	if (mRenderBoxes) mUcb.render(aViewProjection);
}

pbd::gpu_list<4> waterfall::scalar_particle_velocities()
{
	auto result = pbd::gpu_list<4>().request_length(mParticles.requested_length()).set_length(mParticles.length());
	shader_provider::vec3_to_length(mParticles.hidden_list().get<pbd::hidden_particles::id::velocity>().buffer(), result.write().buffer(), mParticles.hidden_list().length());
	return result;
}
