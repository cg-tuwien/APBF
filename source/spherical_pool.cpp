#include "spherical_pool.h"
#include "initialize.h"
#include "measurements.h"
#include "settings.h"

spherical_pool::spherical_pool(const glm::vec3& aCenter, float aPoolRadius, float aParticleRadius) :
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
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_sphere_shape(mParticles, aCenter, aPoolRadius, aParticleRadius);
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.length(), aParticleRadius * static_cast<float>(KERNEL_SCALE), 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::target_radius>().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::boundariness>().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence(mFluid.get<pbd::fluid::id::boundary_distance>().write().buffer(), mFluid.length(), static_cast<uint32_t>(aParticleRadius * POS_RESOLUTION), 0);

	mVelocityHandling .set_data(&mParticles                           ).set_acceleration(glm::vec3(0, -10, 0));
	mSpreadKernelWidth.set_data(&mFluid, &mNeighborsFluid             );
	mIncompressibility.set_data(&mFluid, &mNeighborsFluid             );
	mSphereCollision  .set_data(&mParticles                           ).set_sphere(aCenter, aPoolRadius, true);
	mUpdateTransfers  .set_data(&mFluid, &mNeighborsFluid, &mTransfers);
	mParticleTransfer .set_data(&mFluid, &mTransfers                  );

	mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::kernel_width>(), &mNeighborsFluid);
#if NEIGHBORHOOD_TYPE == 1
	mNeighborhoodFluid.set_position_range(aMin, aMax, 4u);
#endif
	mNeighborhoodFluid.set_range_scale(pbd::settings::baseKernelWidthOnBoundaryDistance ? 1.0f : 1.5f);
	mTimeMachine.set_max_keyframes(4).set_keyframe_interval(120).enable();
	shader_provider::end_recording();
	mRenderBoxes = true;
}

void spherical_pool::update(float aDeltaTime)
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
			mSphereCollision.apply();
			mIncompressibility.apply();
		}

		//if (pbd::settings::merge || pbd::settings::split || pbd::settings::baseKernelWidthOnTargetRadius || pbd::settings::color == 1 || pbd::settings::color == 2) // TODO reactivate?
		{
			mUpdateTransfers.apply();
		}

		mTimeMachine.save_state();
	}
}

pbd::particles& spherical_pool::particles()
{
	return mParticles;
}

pbd::fluid& spherical_pool::fluid()
{
	return mFluid;
}

pbd::neighbors& spherical_pool::neighbors()
{
	return mNeighborsFluid;
}

void spherical_pool::handle_input(const glm::mat4& aInverseViewProjection, const glm::vec3& aCameraPos) {}

void spherical_pool::render(const glm::mat4& aViewProjection) {}

pbd::gpu_list<4> spherical_pool::scalar_particle_velocities()
{
	auto result = pbd::gpu_list<4>().request_length(mParticles.requested_length()).set_length(mParticles.length());
	shader_provider::vec3_to_length(mParticles.hidden_list().get<pbd::hidden_particles::id::velocity>().buffer(), result.write().buffer(), mParticles.hidden_list().length());
	return result;
}
