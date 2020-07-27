#include <gvk.hpp>
#include <imgui.h>
#include <random>
#include "shader_provider.h"
#include "gpu_list.h"

#ifdef _DEBUG
#include "Test.h"
#endif

class apbf : public gvk::invokee
{
	struct particle {
	    glm::vec4 mOriginalPositionRand;
	    glm::vec4 mCurrentPositionRadius;
	};

	struct application_data {
		glm::mat4 mViewMatrix;
		glm::mat4 mProjMatrix;
		glm::vec4 mTime;
	};

public: // v== gvk::invokee overrides which will be invoked by the framework ==v
	apbf(avk::queue& aQueue)
		: mQueue{ &aQueue }
	{
		shader_provider::set_queue(aQueue);
	}

	void initialize() override
	{
		using namespace avk;
		using namespace gvk;

#ifdef _DEBUG
		pbd::test::test_quick();
#endif
		auto* mainWnd = context().main_window();
		const auto framesInFlight = mainWnd->number_of_frames_in_flight();
		
		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = context().create_descriptor_cache();

		// Create the camera and buffers that will contain camera data:
		mQuakeCam.set_translation({ 0.0f, 0.0f, 0.0f });
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), mainWnd->aspect_ratio(), 0.5f, 500.0f);
		current_composition()->add_element(mQuakeCam);
		for (window::frame_id_t i = 0; i < framesInFlight; ++i) {
			mCameraDataBuffer.emplace_back(context().create_buffer(
				memory_usage::host_coherent, {},
				uniform_buffer_meta::create_from_data(application_data{})
			));
		}
		
		// Create particles in a regular grid with random radius:
		std::vector<particle> testParticles;
		std::default_random_engine generator;
		generator.seed(186);
		std::uniform_real_distribution<float> randomFrequency(0.5f, 3.0f);
		std::uniform_real_distribution<float> randomRadius(0.01f, 0.1f);
		const float dist = 2.0f;
		for (int x = 0; x < 60; ++x) {
			for (int y = 0; y < 50; ++y) {
				for (int z = 0; z < 40; ++z) {
					const auto pos = glm::vec3{ x, y, -z - 3.0f };
					testParticles.emplace_back(particle{
						glm::vec4{ pos, randomFrequency(generator) },
						glm::vec4{ pos, randomRadius(generator) },
					});
				}
			}
		}
		mNumParticles = static_cast<uint32_t>(testParticles.size());
		
		// Alloc buffers and ray tracing acceleration structures (one for each frame in flight), fill only mParticlesBuffers:
		for (window::frame_id_t i = 0; i < framesInFlight; ++i) {
			auto& pb = mParticlesBuffer.emplace_back(context().create_buffer(
				memory_usage::device, {}, 
				storage_buffer_meta::create_from_data(testParticles),
				instance_buffer_meta::create_from_data(testParticles),
				vertex_buffer_meta::create_from_data(testParticles)
			));
			pb->fill(testParticles.data(), 0, sync::wait_idle());
			
			mAabbsBuffer.emplace_back(context().create_buffer(
				memory_usage::device, {},
				storage_buffer_meta::create_from_size(sizeof(aabb) * mNumParticles),
				aabb_buffer_meta::create_from_num_elements(mNumParticles)
			));

			auto& blas = mBottomLevelAS.emplace_back(context().create_bottom_level_acceleration_structure({ acceleration_structure_size_requirements::from_aabbs(mNumParticles) }, true));
			mTopLevelAS.emplace_back(context().create_top_level_acceleration_structure(mNumParticles, true));
			auto& gi = mGeometryInstances.emplace_back();
			for (auto& p : testParticles) {
				gi.push_back(context().create_geometry_instance(blas));
			}
		}
		
		// Load a sphere model for drawing a single particle:
		auto sphere = model_t::load_from_file("assets/sphere.obj");
		std::tie(mSphereVertexBuffer, mSphereIndexBuffer) = create_vertex_and_index_buffers( make_models_and_meshes_selection(sphere, 0) );
		
		// Create a graphics pipeline for drawing the particles that uses instanced rendering:
		mGraphicsPipelineInstanced = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/instanced.vert"), 
			fragment_shader("shaders/red.frag"),
			// Declare the vertex input to the shaders:
			vertex_input_location(0, glm::vec3{}).from_buffer_at_binding(0), // Declare that positions shall be read from the attached vertex buffer at binding 0,
			                                                                 // and that we are going to access it in shaders via layout (location = 0)
			instance_input_location(1, &particle::mCurrentPositionRadius).from_buffer_at_binding(1), // Stream instance data from the buffer at binding 1
			context().create_renderpass({
					attachment::declare(format_from_window_color_buffer(mainWnd), on_load::clear,   color(0),         on_store::store),
					attachment::declare(format_from_window_depth_buffer(mainWnd), on_load::clear,   depth_stencil(),  on_store::dont_care)
				},
				[](renderpass_sync& aRpSync){
					// Synchronize with everything that comes BEFORE:
					if (aRpSync.is_external_pre_sync()) {
						aRpSync.mSourceStage                    = pipeline_stage::compute_shader;
						aRpSync.mSourceMemoryDependency         = memory_access::shader_buffers_and_images_write_access;
						aRpSync.mDestinationStage               = pipeline_stage::vertex_input;
						aRpSync.mDestinationMemoryDependency    = memory_access::any_vertex_input_read_access;
					}
					// Synchronize with everything that comes AFTER:
					if (aRpSync.is_external_post_sync()) {
						aRpSync.mSourceStage                    = pipeline_stage::color_attachment_output;
						aRpSync.mDestinationStage               = pipeline_stage::color_attachment_output;
						aRpSync.mSourceMemoryDependency         = memory_access::color_attachment_write_access;
						aRpSync.mDestinationMemoryDependency    = memory_access::color_attachment_write_access;
					}
				}
			),
			// Further config for the pipeline:
			cfg::viewport_depth_scissors_config::from_framebuffer(mainWnd->backbuffer_at_index(0)), // Set to the dimensions of the main window
			binding(0, 0, mCameraDataBuffer[0])
		);

		//std::vector<glm::vec4> four = { glm::vec4{0, 0, 0, 0}, glm::vec4{0, 3, 0, 0}, glm::vec4{3, 0, 0, 0}, glm::vec4{0, 0, 3, 0} };
		//mTest = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(four));
		//mTest->fill(four.data(), 0, sync::wait_idle());
		
		// Create a graphics pipeline for drawing the particles that uses point primitives:
		mGraphicsPipelinePoint = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/point.vert"), 
			fragment_shader("shaders/red.frag"),
			cfg::polygon_drawing::config_for_points(), cfg::rasterizer_geometry_mode::rasterize_geometry, cfg::culling_mode::disabled,
			// Declare the vertex input to the shaders:
			//vertex_input_location(0, 0, vk::Format::eR16G16B16A16Sfloat, 0).from_buffer_at_binding(0), // Stream particle positions from buffer at index 0
			vertex_input_location(0, &particle::mCurrentPositionRadius).from_buffer_at_binding(0),
			context().create_renderpass({
					attachment::declare(format_from_window_color_buffer(mainWnd), on_load::clear,   color(0),         on_store::store),
					attachment::declare(format_from_window_depth_buffer(mainWnd), on_load::clear,   depth_stencil(),  on_store::dont_care)
				},
				[](renderpass_sync& aRpSync){
					// Synchronize with everything that comes BEFORE:
					if (aRpSync.is_external_pre_sync()) {
						aRpSync.mSourceStage                    = pipeline_stage::compute_shader;
						aRpSync.mSourceMemoryDependency         = memory_access::shader_buffers_and_images_write_access;
						aRpSync.mDestinationStage               = pipeline_stage::vertex_input;
						aRpSync.mDestinationMemoryDependency    = memory_access::any_vertex_input_read_access;
					}
					// Synchronize with everything that comes AFTER:
					if (aRpSync.is_external_post_sync()) {
						aRpSync.mSourceStage                    = pipeline_stage::color_attachment_output;
						aRpSync.mDestinationStage               = pipeline_stage::color_attachment_output;
						aRpSync.mSourceMemoryDependency         = memory_access::color_attachment_write_access;
						aRpSync.mDestinationMemoryDependency    = memory_access::color_attachment_write_access;
					}
				}
			),
			// Further config for the pipeline:
			cfg::viewport_depth_scissors_config::from_framebuffer(mainWnd->backbuffer_at_index(0)), // Set to the dimensions of the main window
			binding(0, 0, mCameraDataBuffer[0])
		);

		// Get hold of the "ImGui Manager" and add a callback that draws UI elements:
		auto imguiManager = current_composition()->element_by_type<imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([this](){
		        ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

				static std::vector<float> values;
				values.push_back(1000.0f / ImGui::GetIO().Framerate);
		        if (values.size() > 90) {
			        values.erase(values.begin());
		        }
	            ImGui::PlotLines("ms/frame", values.data(), values.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 100.0f));

				ImGui::Separator();

				ImGui::TextColored(ImVec4(0.f, 0.8f, 0.5f, 1.0f), "Rendering:");
				static const char* const sRenderingMethods[] = {"Instanced Spheres", "Points"};
				ImGui::Combo("Rendering Method", &mRenderingMethod, sRenderingMethods, IM_ARRAYSIZE(sRenderingMethods));

				ImGui::Separator();

				ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.0f, 1.0f), "Simulation:");

				ImGui::Separator();

		        ImGui::End();
			});
		}
	}

	void update() override
	{
		using namespace gvk;

		if (input().key_pressed(key_code::f1)) {
			auto imguiManager = current_composition()->element_by_type<imgui_manager>();
			if (mQuakeCam.is_enabled()) {
				mQuakeCam.disable();
				if (nullptr != imguiManager) { imguiManager->enable_user_interaction(true); }
			}
			else {
				mQuakeCam.enable();
				if (nullptr != imguiManager) { imguiManager->enable_user_interaction(false); }
			}
		}
		
		// On Esc pressed,
		if (input().key_pressed(key_code::escape)) {
			// stop the current composition:
			current_composition()->stop();
		}
	}

	void render() override
	{
		using namespace avk;
		using namespace gvk;

		auto* mainWnd = context().main_window();
		const auto ifi = mainWnd->current_in_flight_index();

		application_data cd{ mQuakeCam.view_matrix(), mQuakeCam.projection_matrix(), glm::vec4{ time().time_since_start(), time().delta_time(), 0.f, 0.f } };
		mCameraDataBuffer[ifi]->fill(&cd, 0, sync::not_required());

		shader_provider::start_recording();

		// COMPUTE

		shader_provider::roundandround(mCameraDataBuffer[ifi], mParticlesBuffer[ifi], mAabbsBuffer[ifi], mNumParticles);
		mParticlesBuffer[ifi]->meta<storage_buffer_meta>().num_elements();

		shader_provider::end_recording();
		
		// Get a command pool to allocate command buffers from:
		auto& commandPool = context().get_command_pool_for_single_use_command_buffers(*mQueue);

		// Create a command buffer and render into the *current* swap chain image:
		auto cmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		cmdBfr->begin_recording();

		// BUILD ACCELERATION STRUCTURES

		cmdBfr->establish_global_memory_barrier(
			pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::acceleration_structure_build,
			memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::acceleration_structure_any_access
		);
		mBottomLevelAS[ifi]->build(mAabbsBuffer[ifi], {}, sync::with_barriers_into_existing_command_buffer(cmdBfr, {}, {}));

		//cmdBfr->establish_global_memory_barrier(
		//	pipeline_stage::acceleration_structure_build,       /* -> */ pipeline_stage::acceleration_structure_build,
		//	memory_access::acceleration_structure_write_access, /* -> */ memory_access::acceleration_structure_any_access
		//);
		//mTopLevelAS[ifi]->build(mGeometryInstances[ifi], {}, sync::with_barriers_into_existing_command_buffer(cmdBfr, {}, {})); // TODO: <--- crashes here. There must be a bug hiding somewhere.

		// GRAPHICS

		if (0 == mRenderingMethod) {
			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelineInstanced->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(mGraphicsPipelineInstanced);
			cmdBfr->bind_descriptors(mGraphicsPipelineInstanced->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_indexed(*mSphereIndexBuffer, mNumParticles, 0u, 0u, 0u, *mSphereVertexBuffer, *mParticlesBuffer[ifi]);
		}
		else {
			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelinePoint->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(mGraphicsPipelinePoint);
			cmdBfr->bind_descriptors(mGraphicsPipelinePoint->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_vertices(*mParticlesBuffer[ifi]);
		}
		
		cmdBfr->end_render_pass();
		cmdBfr->end_recording();

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto& imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();
		
		// Submit the draw call and take care of the command buffer's lifetime:
		mQueue->submit(cmdBfr, imageAvailableSemaphore);
		mainWnd->handle_lifetime(std::move(cmdBfr));
	}

	void finalize() override
	{
		pbd::gpu_list<4>::cleanup();
		pbd::gpu_list<12>::cleanup();
	}


private: // v== Member variables ==v

	avk::queue* mQueue;
	avk::descriptor_cache mDescriptorCache;

	gvk::quake_camera mQuakeCam;
	std::vector<avk::buffer> mCameraDataBuffer;

	uint32_t mNumParticles;
	std::vector<avk::buffer> mParticlesBuffer;
	std::vector<avk::buffer> mAabbsBuffer;
	avk::buffer mSphereVertexBuffer;
	avk::buffer mSphereIndexBuffer;

	std::vector<avk::bottom_level_acceleration_structure> mBottomLevelAS;
	std::vector<avk::top_level_acceleration_structure> mTopLevelAS;
	std::vector<std::vector<avk::geometry_instance>> mGeometryInstances;

	//avk::compute_pipeline mComputePipeline;
	avk::graphics_pipeline mGraphicsPipelineInstanced;
	avk::graphics_pipeline mGraphicsPipelinePoint;
	avk::ray_tracing_pipeline mRayTracingPipeline;
	
	std::vector<avk::image_view> mOffscreenImages;

	//avk::buffer mTest;
	
	// Settings from the UI:
	int mRenderingMethod = 0;
	
}; // class apbf

int main() // <== Starting point ==
{
	using namespace avk;
	using namespace gvk;

	try {
		auto mainWnd = context().create_window("APBF");
		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->set_additional_back_buffer_attachments({ 
			attachment::declare(vk::Format::eD32Sfloat, on_load::clear, depth_stencil(), on_store::dont_care)
		});
		mainWnd->set_presentaton_mode(presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(3u);
		mainWnd->open();

		auto& singleQueue = context().create_queue({}, queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->add_queue_family_ownership(singleQueue);
		mainWnd->set_present_queue(singleQueue);
		
		auto app = apbf(singleQueue);
		auto ui = imgui_manager(singleQueue);

		start(
			application_name("APBF"),
			required_device_extensions()
				.add_extension(VK_KHR_RAY_TRACING_EXTENSION_NAME)
				.add_extension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)
				.add_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
				.add_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
				.add_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
				.add_extension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME),
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