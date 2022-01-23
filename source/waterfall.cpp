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
	mFluid.get<pbd::fluid::id::particle>() = pbd::initialize::add_box_shape(mParticles, aMin, aMax, aRadius);
	mFluid.set_length(mFluid.get<pbd::fluid::id::particle>().length());
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::kernel_width     >().write().buffer(), mFluid.length(), aRadius * static_cast<float>(KERNEL_SCALE), 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::target_radius    >().write().buffer(), mFluid.length(), aRadius, 0);
	shader_provider::write_sequence_float(mFluid.get<pbd::fluid::id::boundariness     >().write().buffer(), mFluid.length(), 1, 0);
	shader_provider::write_sequence      (mFluid.get<pbd::fluid::id::boundary_distance>().write().buffer(), mFluid.length(), static_cast<uint32_t>(aRadius * POS_RESOLUTION), 0);

	// top pool
	auto wallThickness = 4.0f;
	auto wMin = aMin - wallThickness;
	auto wMax = aMax + wallThickness + glm::vec3(0, aRadius * 2, 0);
	mUcb.add_box(wMin, glm::vec3(wMin.x + wallThickness, wMax.y, wMax.z));
	mUcb.add_box(wMin, glm::vec3(wMax.x, wMin.y + wallThickness, wMax.z));
	mUcb.add_box(glm::vec3(wMax.x - wallThickness, wMin.y, wMin.z), wMax);
	mUcb.add_box(glm::vec3(wMin.x, wMax.y - wallThickness, wMin.z), wMax);
	// bottom pool
	auto shift = wMin - wMax;
	shift.z = 0;
	mUcb.add_box(shift + wMin, glm::vec3(wMin.x + wallThickness, wMax.y, wMax.z) + shift);
	mUcb.add_box(shift + wMin, glm::vec3(wMax.x, wMin.y + wallThickness, wMax.z) + shift);
	mUcb.add_box(shift + glm::vec3(wMax.x - wallThickness, wMin.y, wMin.z), wMax + shift);
#if DIMENSIONS > 2
	// top pool
	mUcb.add_box(wMin, glm::vec3(wMax.x, wMax.y, wMin.z + wallThickness));
	mUcb.add_box(glm::vec3(wMin.x, wMin.y, wMax.z - wallThickness), wMax);
	// bottom pool
	mUcb.add_box(shift + wMin, glm::vec3(wMax.x, wMax.y, wMin.z + wallThickness) + shift);
	mUcb.add_box(shift + glm::vec3(wMin.x, wMin.y, wMax.z - wallThickness), wMax + shift);
#endif

	mVelocityHandling .set_data(&mParticles                                  ).set_acceleration(glm::vec3(0, -10, 0));
	mSpreadKernelWidth.set_data(&mFluid, &mNeighborsFluid                    );
	mIncompressibility.set_data(&mFluid, &mNeighborsFluid                    );
	mBoxCollision     .set_data(&mParticles, &mUcb.box_min(), &mUcb.box_max());
	mUpdateTransfers  .set_data(&mFluid, &mNeighborsFluid, &mTransfers       );
	mParticleTransfer .set_data(&mFluid,                   &mTransfers       );
	mSaveParticleInfo .set_data(&mFluid, &mNeighborsFluid, &mTransfers       ).set_boxes(&mUcb.box_min(), &mUcb.box_max());

	mNeighborhoodFluid.set_data(&mFluid.get<pbd::fluid::id::particle>(), &mFluid.get<pbd::fluid::id::kernel_width>(), &mNeighborsFluid);
#if NEIGHBORHOOD_TYPE == 1
	mNeighborhoodFluid.set_position_range(aMin, aMax, 4u);
#endif
	mTimeMachine.set_max_keyframes(4).set_keyframe_interval(120).enable(TIME_MACHINE_ENABLED_AT_START);
	shader_provider::end_recording();
	pbd::settings::smallestTargetRadius = aRadius;
	mMaxExpectedBoundaryDistance = glm::compMin(aMax - aMin) / 2.0f;  // TODO if DIMENSIONS < 3 ignore third dimension
	mViewBoxMax = glm::vec2(wMax) + (mMaxExpectedBoundaryDistance * 0.1f);
	mViewBoxMin = 2.0f * glm::vec2(wMin) - mViewBoxMax;
}

waterfall::waterfall(const glm::vec3& aMin, const glm::vec3& aMax, gvk::camera& aCamera, float aRadius) :
	waterfall(aMin, aMax, aRadius)
{
#if DIMENSIONS == 2
	auto pos   = aMin + glm::normalize(glm::vec3(0, 0, 1)) * glm::distance(aMin, aMax) * 1.5f;
	auto focus = aMin;
#else
	auto pos   = aMin + glm::vec3(0, 0, aMax.z / 2.0f) + glm::normalize(glm::vec3(-0.3, 0.3, 1)) * glm::distance(aMin, aMax) * 1.5f;
	auto focus = aMin + glm::vec3(0, -aMax.y / 2.0f, 0);
#endif
	aCamera.set_translation(pos);
	aCamera.set_rotation(glm::conjugate(glm::quat(glm::lookAt(pos, focus, glm::vec3(0, 1, 0)))));
}

void waterfall::update(float aDeltaTime)
{
	mDeltaTime = FIXED_TIME_STEP == 0 ? aDeltaTime : FIXED_TIME_STEP;
	if (!mTimeMachine.step_forward()) {
		mVelocityHandling.apply(mDeltaTime);

		if (!pbd::settings::basicPbf && (pbd::settings::merge || pbd::settings::split)) {
			mParticleTransfer.apply(mDeltaTime);
		}

		if (!pbd::settings::basicPbf && pbd::settings::baseKernelWidthOnBoundaryDistance) {
			shader_provider::uint_to_float_with_indexed_lower_bound(mFluid.get<pbd::fluid::id::boundary_distance>().buffer(), mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.get<pbd::fluid::id::particle>().index_buffer(), mFluid.get<pbd::fluid::id::particle>().hidden_list().get<pbd::hidden_particles::id::radius>().buffer(), mFluid.length(), pbd::settings::targetRadiusScaleFactor / static_cast<float>(POS_RESOLUTION), static_cast<float>(KERNEL_SCALE), pbd::settings::kernelWidthAdaptionSpeed);
			//shader_provider::uint_to_float_but_gradual(mFluid.get<pbd::fluid::id::boundary_distance>().buffer(), mFluid.get<pbd::fluid::id::kernel_width>().write().buffer(), mFluid.length(), pbd::settings::targetRadiusScaleFactor / POS_RESOLUTION, pbd::settings::kernelWidthAdaptionSpeed, pbd::settings::smallestTargetRadius * KERNEL_SCALE);
		}

		measurements::record_timing_interval_start("Neighborhood");
		mNeighborhoodFluid.set_range_scale(pbd::settings::basicPbf || pbd::settings::baseKernelWidthOnBoundaryDistance ? 1.0f : 1.5f);
		mNeighborhoodFluid.apply();
		measurements::record_timing_interval_end("Neighborhood");

		if (!pbd::settings::basicPbf && !pbd::settings::baseKernelWidthOnBoundaryDistance) {
			mSpreadKernelWidth.apply();
		}

		measurements::record_timing_interval_start("Constraint Solver");
		for (auto i = 0; i < pbd::settings::solverIterations; i++) {
			mBoxCollision.apply();
			mIncompressibility.apply();
		}
		measurements::record_timing_interval_end("Constraint Solver");

		//if (pbd::settings::merge || pbd::settings::split || pbd::settings::baseKernelWidthOnTargetRadius || pbd::settings::color == 1 || pbd::settings::color == 2) // TODO reactivate?
		if (!pbd::settings::basicPbf)
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
	static auto svgId = 0u;
	if (gvk::input().key_pressed(gvk::key_code::g)) {
		shader_provider::start_recording();
		mSaveParticleInfo.save_as_svg(svgId++, mViewBoxMin, mViewBoxMax, pbd::settings::particleRenderScale, mMaxExpectedBoundaryDistance);
		shader_provider::end_recording();
	}

	if (pbd::settings::renderBoxes) mUcb.handle_input(aInverseViewProjection, aCameraPos);
}

void waterfall::render(const glm::mat4& aViewProjection)
{
	if (pbd::settings::renderBoxes) mUcb.render(aViewProjection);
}

pbd::gpu_list<4> waterfall::scalar_particle_velocities()
{
	auto result = pbd::gpu_list<4>().request_length(mParticles.requested_length()).set_length(mParticles.length());
	shader_provider::vec3_to_length(mParticles.hidden_list().get<pbd::hidden_particles::id::velocity>().buffer(), result.write().buffer(), mParticles.hidden_list().length());
	return result;
}

float waterfall::max_expected_boundary_distance()
{
	return mMaxExpectedBoundaryDistance;
}
