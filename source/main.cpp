#include "../shaders/cpu_gpu_shared_config.h"
#include <gvk.hpp>
#include <imgui.h>
#include "shader_provider.h"
#include "measurements.h"
#include "settings.h"
#include "randomParticles.h"

#ifdef _DEBUG
#include "Test.h"
#endif
#include "neighborhood_brute_force.h"
#include "neighborhood_rtx.h"
#include "neighborhood_binary_search.h"
#include "neighborhood_green.h"

class neighborhood : public gvk::invokee
{
	struct camera_data {
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
	neighborhood(avk::queue& aQueue)
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

		shader_provider::start_recording();
		mScene.init(PARTICLE_COUNT, AREA_MIN, AREA_MAX);
		mNeighborhoodRtx.init();
		auto& range = mScene.particles().hidden_list().get<pbd::hidden_particles::id::radius>();
		mNeighborsBruteForce  .request_length(NEIGHBOR_LIST_MAX_LENGTH);
		mNeighborsRtx         .request_length(NEIGHBOR_LIST_MAX_LENGTH);
		mNeighborsGreen       .request_length(NEIGHBOR_LIST_MAX_LENGTH);
		mNeighborsBinarySearch.request_length(NEIGHBOR_LIST_MAX_LENGTH);
		mNeighborhoodBruteForce  .set_data(&mScene.particles(), &range, &mNeighborsBruteForce  ).set_range_scale(1.0f);
		mNeighborhoodRtx         .set_data(&mScene.particles(), &range, &mNeighborsRtx         ).set_range_scale(1.0f);
		mNeighborhoodGreen       .set_data(&mScene.particles(), &range, &mNeighborsGreen       ).set_range_scale(1.0f).set_position_range(AREA_MIN, AREA_MAX, GREEN_RESOLUTION_LOG_2);
		mNeighborhoodBinarySearch.set_data(&mScene.particles(), &range, &mNeighborsBinarySearch).set_range_scale(1.0f);
		shader_provider::end_recording();

		// Create the camera and buffers that will contain camera data:
		mQuakeCam.set_translation({ 0.0f, 0.0f, 0.0f });
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), mainWnd->aspect_ratio(), 0.5f, 500.0f);
		mQuakeCam.set_move_speed(30.0f);
		mQuakeCam.disable();
		gvk::current_composition()->add_element(mQuakeCam);
		for (gvk::window::frame_id_t i = 0; i < framesInFlight; ++i) {
			mCameraDataBuffer.emplace_back(gvk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::uniform_buffer_meta::create_from_data(camera_data{})
			));
		}

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
				ImGui::Text("%.3f ms/Neighborhood", measurements::get_timing_interval_in_ms("neighborhood"));
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

				static std::vector<float> values;
				values.push_back(1000.0f / ImGui::GetIO().Framerate);
				if (values.size() > 90) {
					values.erase(values.begin());
				}
				ImGui::PlotLines("ms/frame", values.data(), static_cast<int>(values.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 100.0f));
				
				ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.5f, 1.0f), "Neighbor Count:");
				ImGui::Text("%d Brute Force"  , mNeighborsBruteForce  .empty() ? 0 : measurements::async_read_uint("bf" , mNeighborsBruteForce  .length()));
				ImGui::Text("%d RTX"          , mNeighborsRtx         .empty() ? 0 : measurements::async_read_uint("rtx", mNeighborsRtx         .length()));
				ImGui::Text("%d Green"        , mNeighborsGreen       .empty() ? 0 : measurements::async_read_uint("gr" , mNeighborsGreen       .length()));
				ImGui::Text("%d Binary Search", mNeighborsBinarySearch.empty() ? 0 : measurements::async_read_uint("bs" , mNeighborsBinarySearch.length()));

				ImGui::Separator();

				ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.0f, 1.0f), "Computation Time:");
				ImGui::Text("%.3f ms Brute Force"  , measurements::get_timing_interval_in_ms("bf" ));
				ImGui::Text("%.3f ms RTX"          , measurements::get_timing_interval_in_ms("rtx"));
				ImGui::Text("%.3f ms Green"        , measurements::get_timing_interval_in_ms("gr" ));
				ImGui::Text("%.3f ms Binary Search", measurements::get_timing_interval_in_ms("bs" ));

				ImGui::Separator();

				pbd::settings::add_settings_im_gui_entries();

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
			//mFreezeParticleAnimation = !mFreezeParticleAnimation;
		}

		if (gvk::input().key_pressed(gvk::key_code::enter)) {
			//mPerformSingleSimulationStep = true;
		}
		
		// On Esc pressed,
		if (gvk::input().key_pressed(gvk::key_code::escape)) {
			// stop the current composition:
			gvk::current_composition()->stop();
		}
	}

	void render() override
	{
		auto* mainWnd = gvk::context().main_window();
		const auto ifi = mainWnd->current_in_flight_index();

		camera_data cd{
			// Camera's view matrix
			mQuakeCam.view_matrix(),
			// Camera's projection matrix
			mQuakeCam.projection_matrix()
		};
		mCameraDataBuffer[ifi]->fill(&cd, 0, avk::sync::not_required());
		pbd::settings::update_settings_buffer();

		shader_provider::start_recording();
		measurements::record_timing_interval_start("neighborhood");

		measurements::record_timing_interval_start("bf" ); if (pbd::settings::bruteForceActive  ) mNeighborhoodBruteForce  .apply(); measurements::record_timing_interval_end("bf" );
		measurements::record_timing_interval_start("rtx"); if (pbd::settings::rtxActive         ) mNeighborhoodRtx         .apply(); measurements::record_timing_interval_end("rtx");
		measurements::record_timing_interval_start("gr" ); if (pbd::settings::greenActive       ) mNeighborhoodGreen       .apply(); measurements::record_timing_interval_end("gr" );
		measurements::record_timing_interval_start("bs" ); if (pbd::settings::binarySearchActive) mNeighborhoodBinarySearch.apply(); measurements::record_timing_interval_end("bs" );

		measurements::record_timing_interval_end("neighborhood");
		shader_provider::end_recording();

		// GRAPHICS

		shader_provider::start_recording();
		auto& cmdBfr = shader_provider::cmd_bfr();

		measurements::debug_label_start("Rendering", glm::vec4(0, 0.5, 0, 1));

		auto result = &mImages[ifi].mColor;
		auto position = mScene.particles().hidden_list().get<pbd::hidden_particles::id::position>();
		auto radius   = mScene.particles().hidden_list().get<pbd::hidden_particles::id::radius  >();
		auto floatForColor = radius;
		auto neighbors = mNeighborsBruteForce;

		shader_provider::write_sequence(floatForColor.write().buffer(), position.length(), 0u, 0u);
		shader_provider::neighbor_list_to_particle_mask(mScene.particles().index_buffer(), neighbors.buffer(), floatForColor.write().buffer(), neighbors.length(), pbd::settings::focusParticleId);

		auto color1      = glm::vec3(1, 0, 0);
		auto color2      = glm::vec3(0, 0, 1);
		auto color1Float = 0.0f;
		auto color2Float = 1.0f;

		shader_provider::render_particles(mCameraDataBuffer[ifi], mSphereVertexBuffer, mSphereIndexBuffer, position.buffer(), radius.buffer(), floatForColor.buffer(), position.length(), mImages[ifi].mNormal, mImages[ifi].mDepth, mImages[ifi].mColor, static_cast<uint32_t>(mSphereIndexBuffer->meta_at_index<avk::generic_buffer_meta>().num_elements()), color1, color2, color1Float, color2Float, pbd::settings::particleRenderScale);
		blit_image           (          (*result)->get_image(), mainWnd->current_backbuffer()->image_view_at(0)->get_image(), avk::sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));
		copy_image_to_another(mImages[ifi].mDepth->get_image(), mainWnd->current_backbuffer()->image_view_at(1)->get_image(), avk::sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));
		shader_provider::sync_after_transfer();

		measurements::debug_label_end();

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();
		
		shader_provider::end_recording(imageAvailableSemaphore);

		pbd::gpu_list_data::garbage_collection();
	}

	void finalize() override
	{
		pbd::gpu_list_data::cleanup();
		measurements::clean_up_resources();
	}


private: // v== Member variables ==v

	avk::queue* mQueue;

	gvk::quake_camera mQuakeCam;
	std::vector<avk::buffer> mCameraDataBuffer;

	avk::buffer mSphereVertexBuffer;
	avk::buffer mSphereIndexBuffer;
	
	std::vector<images> mImages;

	randomParticles mScene;

	pbd::neighborhood_brute_force   mNeighborhoodBruteForce;
	pbd::neighborhood_rtx           mNeighborhoodRtx;
	pbd::neighborhood_green         mNeighborhoodGreen;
	pbd::neighborhood_binary_search mNeighborhoodBinarySearch;

	pbd::neighbors mNeighborsBruteForce;
	pbd::neighbors mNeighborsRtx;
	pbd::neighbors mNeighborsGreen;
	pbd::neighbors mNeighborsBinarySearch;
	
}; // class neighborhood

int main() // <== Starting point ==
{
	try {
		auto mainWnd = gvk::context().create_window("Neighborhood");
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
		
		auto app = neighborhood(singleQueue);
		auto ui = gvk::imgui_manager(singleQueue);

		start(
			gvk::application_name("Neighborhood"),
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
