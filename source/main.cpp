#include "../shaders/cpu_gpu_shared_config.h"
#include <gvk.hpp>
#include <imgui.h>
#include <random>
#include "shader_provider.h"
#include SCENE_FILENAME
#include "measurements.h"
#include "settings.h"

#ifdef _DEBUG
#include "Test.h"
#endif

class apbf : public gvk::invokee
{
	struct application_data {
		/** Camera's view matrix */
		glm::mat4 mViewMatrix;
		/** Camera's projection matrix */
		glm::mat4 mProjMatrix;
	};

	struct images
	{
		avk::image_view mColor;
		avk::image_view mNormal;
		avk::image_view mDepth;
		avk::image_view mOcclusion;
		avk::image_view mResult;
	};

public: // v== gvk::invokee overrides which will be invoked by the framework ==v
	apbf(avk::queue& aQueue)
		: mQueue{ &aQueue }
	{
		shader_provider::set_queue(aQueue);
	}

	void initialize() override
	{
		shader_provider::set_updater(&mUpdater.emplace());

#ifdef _DEBUG
		pbd::test::test_all();
#endif
		auto* mainWnd = gvk::context().main_window();
		const auto framesInFlight = mainWnd->number_of_frames_in_flight();

		// Create the camera and buffers that will contain camera data:
		mQuakeCam.set_translation({ 0.0f, 0.0f, 0.0f });
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), mainWnd->aspect_ratio(), 0.5f, 500.0f);
		mQuakeCam.set_move_speed(30.0f);
		mQuakeCam.disable();
		gvk::current_composition()->add_element(mQuakeCam);
		for (gvk::window::frame_id_t i = 0; i < framesInFlight; ++i) {
			mCameraDataBuffer.emplace_back(gvk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::uniform_buffer_meta::create_from_data(application_data{})
			));
		}

#if SCENE == 0 || SCENE == 2
		mPool = std::make_unique<SCENE_NAME>(glm::vec3(-40, -10, -80), glm::vec3(40, 30, -40), mQuakeCam, 1.0f);
#elif SCENE == 1
		mPool = std::make_unique<spherical_pool>(glm::vec3(0, 10, -60), 100.0f, mQuakeCam, 1.0f);
#elif SCENE == 3
		mPool = std::make_unique<SCENE_NAME>(glm::vec3(-40, -10, -80) * 2.0f, glm::vec3(40, 30, -40) * 2.0f, glm::vec3(30, 60, -60) * 2.0f, 6.0f, mQuakeCam, 1.0f);
#endif

		for (gvk::window::frame_id_t i = 0; i < framesInFlight; ++i) {
			auto imColor     = gvk::context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,          vk::Format::eR16G16B16A16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_color_attachment | avk::image_usage::shader_storage);
			auto imNormal    = gvk::context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,          vk::Format::eR16G16B16A16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_color_attachment);
			auto imDepth     = gvk::context().create_image(mainWnd->resolution().x, mainWnd->resolution().y, format_from_window_depth_buffer(mainWnd), 1, avk::memory_usage::device, avk::image_usage::general_depth_stencil_attachment | avk::image_usage::input_attachment);
			auto imOcclusion = gvk::context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,                   vk::Format::eR16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_color_attachment);
			auto imResult    = gvk::context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,               vk::Format::eR8G8B8A8Unorm, 1, avk::memory_usage::device, avk::image_usage::general_storage_image);
//			imColor    ->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imNormal   ->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imDepth    ->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imOcclusion->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
//			imResult   ->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
			imColor    ->transition_to_layout();
			imNormal   ->transition_to_layout();
			imDepth    ->transition_to_layout();
			imOcclusion->transition_to_layout();
			imResult   ->transition_to_layout();
			mImages.emplace_back(gvk::context().create_image_view(avk::owned(imColor)), gvk::context().create_image_view(avk::owned(imNormal)), gvk::context().create_image_view(avk::owned(imDepth)), gvk::context().create_image_view(avk::owned(imOcclusion)), gvk::context().create_image_view(avk::owned(imResult)));
		}
		
		// Load a sphere model for drawing a single particle:
		auto sphere = gvk::model_t::load_from_file("assets/icosahedron.obj");
		std::tie(mSphereVertexBuffer, mSphereIndexBuffer) = create_vertex_and_index_buffers( make_models_and_meshes_selection(sphere, 0) );
		
		// Get hold of the "ImGui Manager" and add a callback that draws UI elements:
		auto imguiManager = gvk::current_composition()->element_by_type<gvk::imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([this](){
				ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
//				ImGui::Text("%.3f ms/Simulation Step", measurements::get_timing_interval_in_ms("Simulation Step"));
//				ImGui::Text("%.3f ms/Neighborhood", measurements::get_timing_interval_in_ms("Neighborhood"));
				measurements::add_timing_interval_gui();
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

				static std::vector<float> values;
				values.push_back(1000.0f / ImGui::GetIO().Framerate);
				if (values.size() > 90) {
					values.erase(values.begin());
				}
				ImGui::PlotLines("ms/frame", values.data(), static_cast<int>(values.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 100.0f));
				ImGui::Text("%d Particles", measurements::async_read_uint("particle count", mPool->particles().length()));
				ImGui::Text("%d Neighbor Pairs", mPool->neighbors().empty() ? 0 : measurements::async_read_uint("neighbor count", mPool->neighbors().length()));

				ImGui::Separator();

				ImGui::TextColored(ImVec4(0.f, 0.8f, 0.5f, 1.0f), "Rendering:");
				ImGui::Checkbox("Freeze Particle Animation", &mFreezeParticleAnimation);
				ImGui::Checkbox("Render Boxes", &pbd::settings::renderBoxes);
				ImGui::Checkbox("Ambient Occlusion", &mAddAmbientOcclusion);
				ImGui::Combo("Color", &pbd::settings::color, pbd::settings::colorNames.data(), static_cast<int>(pbd::settings::colorNames.size()));
				ImGui::SliderFloat("Render Scale", &pbd::settings::particleRenderScale, 0.0f, 1.0f, "%.1f");
				ImGui::SliderInt("Render Limit", &pbd::settings::particleRenderLimit, 0, 1000);

				ImGui::Separator();

				ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.0f, 1.0f), "Simulation:");
				ImGui::SliderFloat("Max Delta Time", &mMaxDeltaTime, 0.0f, 0.1f, "%.2f ms");
				pbd::settings::add_apbf_settings_im_gui_entries();
//				ImGui::Separator();
				mImGuiHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

				ImGui::End();
			});
		}
	}

	void update() override
	{
		if (gvk::input().key_pressed(gvk::key_code::f1) || gvk::input().mouse_button_pressed(2) || gvk::input().mouse_button_released(2)) {
			auto imguiManager = gvk::current_composition()->element_by_type<gvk::imgui_manager>();
			if (mQuakeCam.is_enabled()) { mQuakeCam.disable(); if (nullptr != imguiManager) imguiManager->enable_user_interaction(true ); }
			else                        { mQuakeCam. enable(); if (nullptr != imguiManager) imguiManager->enable_user_interaction(false); }
		}

		if (gvk::input().key_pressed(gvk::key_code::space)) {
			mFreezeParticleAnimation = !mFreezeParticleAnimation;
		}

		if (gvk::input().key_pressed(gvk::key_code::enter)) {
			mPerformSingleSimulationStep = true;
		}

		if (gvk::input().key_pressed(gvk::key_code::backspace)) {
			mPool->time_machine().jump_back();
		}

		if (gvk::input().key_pressed(gvk::key_code::t)) {
			mPool->time_machine().toggle_enabled();
		}

		if (gvk::input().key_pressed(gvk::key_code::c)) {
			if (gvk::input().key_down(gvk::key_code::left_shift)) {
				pbd::settings::previousColor();
			}
			else {
				pbd::settings::nextColor();
			}
		}
		
		// On Esc pressed,
		if (gvk::input().key_pressed(gvk::key_code::escape)) {
			// stop the current composition:
			gvk::current_composition()->stop();
		}

		if (!mQuakeCam.is_enabled() && !mImGuiHovered) {
			mPool->handle_input(glm::inverse(mQuakeCam.projection_and_view_matrix()), mQuakeCam.translation());
		}
	}

	void render() override
	{
		auto* mainWnd = gvk::context().main_window();
		const auto ifi = mainWnd->current_in_flight_index();

		static float sTimeSinceStartForAnimation = 0.0f;
		if (!mFreezeParticleAnimation) {
			sTimeSinceStartForAnimation = gvk::time().time_since_start();
		}
		application_data cd{
			// Camera's view matrix
			mQuakeCam.view_matrix(),
			// Camera's projection matrix
			mQuakeCam.projection_matrix()
		};
		mCameraDataBuffer[ifi]->fill(&cd, 0, avk::sync::not_required());
		pbd::settings::update_apbf_settings_buffer();

		shader_provider::start_recording();

		if (!mFreezeParticleAnimation || mPerformSingleSimulationStep) {
			measurements::record_timing_interval_start("Simulation Step");
			mPool->update(std::min(gvk::time().delta_time(), mMaxDeltaTime));
			measurements::record_timing_interval_end("Simulation Step");
		}
		if (mPerformSingleSimulationStep) {
			mPerformSingleSimulationStep = false;
			mFreezeParticleAnimation = true;
		}

		// workaround to have position and radius match floatForColor: apply the particle resorting given by the index list in fluid
		// deletes non-fluid particles from display!
		auto& pos = mPool->particles().hidden_list().get<pbd::hidden_particles::id::position>();
		auto& rad = mPool->particles().hidden_list().get<pbd::hidden_particles::id::radius>();
		auto& idx = mPool->fluid().get<pbd::fluid::id::particle>();
		auto position = pbd::gpu_list<16>().request_length(idx.requested_length()).set_length(idx.length());
		auto radius   = pbd::gpu_list< 4>().request_length(idx.requested_length()).set_length(idx.length());
		shader_provider::copy_scattered_read(pos.buffer(), position.write().buffer(), idx.index_buffer(), idx.length(), 16);
		shader_provider::copy_scattered_read(rad.buffer(),   radius.write().buffer(), idx.index_buffer(), idx.length(),  4);

		// use this instead if floatForColor doesn't have to match (or is not a fluid property):
//		auto position = mPool->particles().hidden_list().get<pbd::hidden_particles::id::position>();
//		auto radius   = mPool->particles().hidden_list().get<pbd::hidden_particles::id::radius>();
		auto transferring = mPool->particles().hidden_list().get<pbd::hidden_particles::id::transferring>();

		pbd::gpu_list<4> floatForColor;
		auto color1 = glm::vec3(0, 0, 1);
		auto color2 = glm::vec3(0.62, 0.96, 0.83);
		auto color1Float = 0.0f;
		auto color2Float = 1.0f;
		auto isUint = false;
		auto isParticleProperty = false;
		auto maxBd = mPool->max_expected_boundary_distance();
		auto minRad = pbd::settings::smallestTargetRadius;
		auto maxRad = pbd::settings::targetRadiusScaleFactor * maxBd / (KERNEL_SCALE + KERNEL_SCALE * pbd::settings::targetRadiusScaleFactor);
//		auto minKer = minRad * KERNEL_SCALE;
//		auto maxKer = pbd::settings::targetRadiusScaleFactor * maxBd;
		switch (pbd::settings::color) {
			case 0: floatForColor = mPool->fluid().get<pbd::fluid::id::boundariness     >();        color2 = glm::vec3(1, 0, 0);                                break;
			case 1: floatForColor = mPool->fluid().get<pbd::fluid::id::boundary_distance>();        color2Float = POS_RESOLUTION * maxBd * 0.8f; isUint = true; break;
			case 2: floatForColor = transferring;                        isParticleProperty = true;                                              isUint = true; break;
//			case 3: floatForColor = mPool->fluid().get<pbd::fluid::id::kernel_width     >();        color1Float = minKer; color2Float = maxKer     ; break;
			case 3: floatForColor = mPool->fluid().get<pbd::fluid::id::kernel_width     >();        color1Float = minRad; color2Float = maxRad     ; break; //same color mapping as radius for better comparability
			case 4: floatForColor = mPool->fluid().get<pbd::fluid::id::target_radius    >();        color1Float = minRad; color2Float = maxRad     ; break;
			case 5: floatForColor = radius;                                                         color1Float = minRad; color2Float = maxRad     ; break;
			case 6: floatForColor = mPool->scalar_particle_velocities(); isParticleProperty = true; color1Float = 0     ; color2Float = minRad * 10; break;
		}
		if (isParticleProperty) {
			auto old = floatForColor;
			shader_provider::copy_scattered_read(old.buffer(), floatForColor.write().buffer(), idx.index_buffer(), idx.length(), 4);
			floatForColor.set_length(mPool->fluid().length());
		}
		if (isUint) {
			shader_provider::uint_to_float(floatForColor.write().buffer(), floatForColor.write().buffer(), floatForColor.write().length(), 1.0f);
		}

		auto& cmdBfr = shader_provider::cmd_bfr();

		// GRAPHICS

		measurements::debug_label_start("Rendering", glm::vec4(0, 0.5, 0, 1));

		auto fragToVS = glm::inverse(mQuakeCam.projection_matrix()) * glm::translate(glm::vec3(-1, -1, 0)) * glm::scale(glm::vec3(2.0f / glm::vec2(mainWnd->resolution()), 1.0f));
		auto result = &mImages[ifi].mColor;
		static auto lengthLimit = pbd::gpu_list<4>().request_length(1); // TODO maybe more elegant solution? Or just remove this debug functionality
		if (pbd::settings::particleRenderLimit != 0) lengthLimit.set_length(pbd::settings::particleRenderLimit);
		auto& particleCount = pbd::settings::particleRenderLimit == 0 ? position.length() : lengthLimit.length();

		shader_provider::render_particles(mCameraDataBuffer[ifi], mSphereVertexBuffer, mSphereIndexBuffer, position.buffer(), radius.buffer(), floatForColor.buffer(), particleCount, mImages[ifi].mNormal, mImages[ifi].mDepth, mImages[ifi].mColor, static_cast<uint32_t>(mSphereIndexBuffer->meta_at_index<avk::generic_buffer_meta>().num_elements()), color1, color2, color1Float, color2Float, pbd::settings::particleRenderScale);
		if (mAddAmbientOcclusion) {
			shader_provider::render_ambient_occlusion(mCameraDataBuffer[ifi], mSphereVertexBuffer, mSphereIndexBuffer, position.buffer(), radius.buffer(), position.length(), mImages[ifi].mNormal, mImages[ifi].mDepth, mImages[ifi].mOcclusion, static_cast<uint32_t>(mSphereIndexBuffer->meta_at_index<avk::generic_buffer_meta>().num_elements()), fragToVS, pbd::settings::particleRenderScale);
			shader_provider::darken_image(mImages[ifi].mOcclusion, mImages[ifi].mColor, mImages[ifi].mResult, 0.7f);
			result = &mImages[ifi].mResult;
		}
		blit_image           (          (*result)->get_image(), mainWnd->current_backbuffer()->image_view_at(0)->get_image(), avk::sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));
		copy_image_to_another(mImages[ifi].mDepth->get_image(), mainWnd->current_backbuffer()->image_view_at(1)->get_image(), avk::sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));
		shader_provider::sync_after_transfer();
		mPool->render(mQuakeCam.projection_and_view_matrix());

		measurements::debug_label_end();
		
		cmdBfr->end_recording();

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();
		
		// Submit the draw call and take care of the command buffer's lifetime:
		mQueue->submit(cmdBfr, imageAvailableSemaphore);
		mainWnd->handle_lifetime(std::move(cmdBfr));

		pbd::gpu_list_data::garbage_collection();
	}

	void finalize() override
	{
		// TODO let all templated classes use the same buffer cache
		pbd::gpu_list_data::cleanup();
		measurements::clean_up_resources();
	}


private: // v== Member variables ==v

	avk::queue* mQueue;

	gvk::quake_camera mQuakeCam;
	std::vector<avk::buffer> mCameraDataBuffer;

	std::unique_ptr<SCENE_NAME> mPool;

	avk::buffer mSphereVertexBuffer;
	avk::buffer mSphereIndexBuffer;

	std::vector<images> mImages;

	bool mImGuiHovered = false;
	
	// Settings from the UI:
	bool mFreezeParticleAnimation = true;
	bool mAddAmbientOcclusion = true;
	bool mPerformSingleSimulationStep = false;
	float mMaxDeltaTime = 0.1f;
	
}; // class apbf

int main() // <== Starting point ==
{
	try {
		auto mainWnd = gvk::context().create_window("APBF");
		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->set_additional_back_buffer_attachments({ 
			avk::attachment::declare(vk::Format::eD32Sfloat, avk::on_load::clear, avk::depth_stencil(), avk::on_store::dont_care)
		});
		mainWnd->set_presentaton_mode(gvk::presentation_mode::fifo);
		mainWnd->set_number_of_concurrent_frames(3u);
		mainWnd->open();

		auto& singleQueue = gvk::context().create_queue({}, avk::queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->add_queue_family_ownership(singleQueue);
		mainWnd->set_present_queue(singleQueue);
		
		auto app = apbf(singleQueue);
		auto ui = gvk::imgui_manager(singleQueue);

		start(
			gvk::application_name("APBF"),
			gvk::required_device_extensions()
				.add_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
				.add_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
				.add_extension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)
				.add_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
				.add_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
				.add_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
				.add_extension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME),
			[](vk::PhysicalDeviceFeatures& deviceFeatures) {
				deviceFeatures.setShaderInt64(VK_TRUE);
			},
			[](vk::PhysicalDeviceVulkan12Features& vulkan12Features) {
				vulkan12Features.setBufferDeviceAddress(VK_TRUE);
			},
			[](vk::PhysicalDeviceRayTracingPipelineFeaturesKHR& aRayTracingFeatures) {
				aRayTracingFeatures.setRayTracingPipeline(VK_TRUE);
			},
			[](vk::PhysicalDeviceRayQueryFeaturesKHR& aRayQueryFeatures) {
				aRayQueryFeatures.setRayQuery(VK_TRUE);
			},
			[](vk::PhysicalDeviceAccelerationStructureFeaturesKHR& aAccelerationStructureFeatures) {
				aAccelerationStructureFeatures.setAccelerationStructure(VK_TRUE);
			},
			mainWnd,
			app,
			ui
		);		
	}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}
	catch (gvk::logic_error&) {}
	catch (gvk::runtime_error&) {}
}
