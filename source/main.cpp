#include <gvk.hpp>
#include <imgui.h>
#include <random>
#include "shader_provider.h"
#include "pool.h"

#ifdef _DEBUG
#include "Test.h"
#endif

#include "../shaders/cpu_gpu_shared_config.h"

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

	struct aligned_aabb
	{
		glm::vec3 mMinBounds;
		glm::vec3 mMaxBounds;
		glm::vec2 _align;
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
		pbd::test::test_all();
#endif
		mPool = std::make_unique<pool>();
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
		for (int x = 0; x < 64; ++x) {
			for (int y = 0; y < 48; ++y) {
				for (int z = 0; z < 32; ++z) {
					const auto pos = glm::vec3{ x, y, -z - 3.0f };
					testParticles.emplace_back(particle{
						glm::vec4{ pos, randomFrequency(generator) },
						glm::vec4{ pos, randomRadius(generator)    },
					});
				}
			}
		}
		mNumParticles = static_cast<uint32_t>(testParticles.size());

#if INST_CENTRIC
		mSingleBlas = context().create_bottom_level_acceleration_structure({ acceleration_structure_size_requirements::from_aabbs(1u) }, false);
		mSingleBlas->build({ VkAabbPositionsKHR{ /* min: */ -1.f, -1.f, -1.f,  /* max: */ 1.f,  1.f,  1.f } });

		std::vector<geometry_instance> geomInstInitData;
		uint32_t customIndex = 0;
		for (const auto& p : testParticles) {
			auto pos = glm::vec3{p.mCurrentPositionRadius.x, p.mCurrentPositionRadius.y, p.mCurrentPositionRadius.z};
			auto scl = glm::vec3{p.mCurrentPositionRadius.w};
			geomInstInitData.push_back(
				context().create_geometry_instance(mSingleBlas)
				.set_transform_column_major(to_array(glm::translate(glm::mat4{1.0f}, pos) * glm::scale(scl)))
					.set_custom_index(customIndex++)
					.set_flags(vk::GeometryInstanceFlagBitsKHR::eForceOpaque)
			);
		}
		auto geomInstInitDataForGpu = convert_for_gpu_usage(geomInstInitData);
#endif
		
		// Alloc buffers and ray tracing acceleration structures (one for each frame in flight), fill only mParticlesBuffers:
		for (window::frame_id_t i = 0; i < framesInFlight; ++i) {
			auto& pb = mParticlesBuffer.emplace_back(context().create_buffer(
				memory_usage::device, {}, 
				storage_buffer_meta::create_from_data(testParticles),
				instance_buffer_meta::create_from_data(testParticles),
				vertex_buffer_meta::create_from_data(testParticles)
			));
			pb->fill(testParticles.data(), 0, sync::wait_idle());

#if BLAS_CENTRIC
			mAabbsBuffer.emplace_back(context().create_buffer(
				memory_usage::device, {},
				storage_buffer_meta::create_from_size(sizeof(aligned_aabb) * mNumParticles),
				aabb_buffer_meta::create_from_num_elements(mNumParticles, sizeof(aligned_aabb))
			));

			// One AABB for every particle:
			auto& blas = mBottomLevelAS.emplace_back(context().create_bottom_level_acceleration_structure({ acceleration_structure_size_requirements::from_aabbs(mNumParticles) }, true));
			std::vector<aabb> temp;
			std::vector<aligned_aabb> tempAligned;
			for (auto& p : testParticles) {
				temp.push_back(aabb{ 
					{{ p.mCurrentPositionRadius.x - p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.y - p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.z - p.mCurrentPositionRadius.w }},
					{{ p.mCurrentPositionRadius.x + p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.y + p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.z + p.mCurrentPositionRadius.w }}
				});
				tempAligned.push_back(aligned_aabb{ 
					glm::vec3{ p.mCurrentPositionRadius.x - p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.y - p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.z - p.mCurrentPositionRadius.w },
					glm::vec3{ p.mCurrentPositionRadius.x + p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.y + p.mCurrentPositionRadius.w, p.mCurrentPositionRadius.z + p.mCurrentPositionRadius.w },
					glm::vec2{}
				});
			}
			blas->build(temp);

			//mAabbsBuffer.back()->fill(tempAligned.data(), 0, sync::wait_idle());
			//blas->build(mAabbsBuffer.back());

			auto& tlas = mTopLevelAS.emplace_back(context().create_top_level_acceleration_structure(1u, true)); // All into one single top-level instance
			tlas->build({ context().create_geometry_instance(blas) });
#else
			static_assert(INST_CENTRIC);
			
			auto& gib = mGeometryInstanceBuffers.emplace_back(context().create_buffer(
				memory_usage::device, {},
				storage_buffer_meta::create_from_data(geomInstInitDataForGpu),
				geometry_instance_buffer_meta::create_from_data(geomInstInitDataForGpu)
			));
			gib->fill(geomInstInitDataForGpu.data(), 1, sync::wait_idle());
			
			auto& tlas = mTopLevelAS.emplace_back(context().create_top_level_acceleration_structure(mNumParticles, true)); // One top level instance per particle
			tlas->build(gib);
#endif
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
		
		// Create a graphics pipeline for drawing the particles that uses instanced rendering:
		mGraphicsPipelineInstanced2 = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/instanced2.vert"), 
			fragment_shader("shaders/red.frag"),
			// Declare the vertex input to the shaders:
			vertex_input_location(0, glm::vec3{}).from_buffer_at_binding(0), // Declare that positions shall be read from the attached vertex buffer at binding 0,
			                                                                 // and that we are going to access it in shaders via layout (location = 0)
			instance_input_location(1, glm::ivec4{}).from_buffer_at_binding(1), // Stream instance data from the buffer at binding 1
			instance_input_location(2, 0.0f).from_buffer_at_binding(2),         // Stream instance data from the buffer at binding 1
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

		// Create offscreen image views to ray-trace into, one for each frame in flight:
		const auto wdth = mainWnd->resolution().x;
		const auto hght = mainWnd->resolution().y;
		const auto frmt = format_from_window_color_buffer(mainWnd);
		for (window::frame_id_t i = 0; i < framesInFlight; ++i) {
			auto& imv = mOffscreenImageViews.emplace_back(context().create_image_view(context().create_image(wdth, hght, frmt, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)));
			imv->get_image().transition_to_layout();
			assert((mOffscreenImageViews.back()->config().subresourceRange.aspectMask & vk::ImageAspectFlagBits::eColor) == vk::ImageAspectFlagBits::eColor);
		}
		
		// Create ray tracing pipeline to render the particles using ray tracing:
		mRayTracingPipeline = context().create_ray_tracing_pipeline_for(
			define_shader_table(
				"shaders/ray_tracing/rt_trace_rays.rgen",
				procedural_hit_group::create_with_rint_and_rchit("shaders/ray_tracing/rt_stupid_intersection.rint", "shaders/ray_tracing/rt_hit_green.rchit"),
				"shaders/ray_tracing/rt_hit_miss.rmiss"
			),
			context().get_max_ray_tracing_recursion_depth(),
			// Define push constants and descriptor bindings:
			binding(0, 0, mCameraDataBuffer[0]),
			binding(1, 0, mParticlesBuffer[0]),
#if BLAS_CENTRIC
			binding(1, 1, mAabbsBuffer[0]->as_storage_buffer()),
#else 
			binding(1, 1, mGeometryInstanceBuffers[0]->as_storage_buffer()),
#endif
			binding(2, 0, mOffscreenImageViews[0]->as_storage_image()), // Just take any, this is just to define the layout
			binding(2, 1, mTopLevelAS[0])                               // Just take any, this is just to define the layout
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
				static const char* const sRenderingMethods[] = {"Instanced Spheres", "Points", "Ray Tracing", "Fluid"};
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

		mPool->update(gvk::time().delta_time());

		// COMPUTE

#if BLAS_CENTRIC
		shader_provider::roundandround(mCameraDataBuffer[ifi], mParticlesBuffer[ifi], mAabbsBuffer[ifi], mNumParticles);
#else
		shader_provider::roundandround(mCameraDataBuffer[ifi], mParticlesBuffer[ifi], mGeometryInstanceBuffers[ifi], mNumParticles);
#endif
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
		
#if BLAS_CENTRIC
		//mBottomLevelAS[ifi]->update(mAabbsBuffer[ifi], {}, sync::with_barriers_into_existing_command_buffer(cmdBfr, {}, {}));
		//// TODO: switch to     ^ update() as soon as the validation layer errors have been fixed: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2066

		//cmdBfr->establish_global_memory_barrier(
		//	pipeline_stage::acceleration_structure_build,       /* -> */ pipeline_stage::acceleration_structure_build,
		//	memory_access::acceleration_structure_write_access, /* -> */ memory_access::acceleration_structure_any_access
		//);
		//cmdBfr->establish_global_memory_barrier(pipeline_stage::all_commands, pipeline_stage::all_commands,	memory_access::any_write_access, memory_access::any_read_access);

		mTopLevelAS[ifi]->update({ context().create_geometry_instance(mBottomLevelAS[ifi]) }, {}, sync::with_barriers_into_existing_command_buffer(cmdBfr, {}, {}));
		// TODO: switch to  ^ update() as soon as the validation layer errors have been fixed: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2066
#else
		static_assert(INST_CENTRIC);
		mTopLevelAS[ifi]->build(mGeometryInstanceBuffers[ifi], {}, sync::with_barriers_into_existing_command_buffer(cmdBfr, {}, {}));
#endif

		cmdBfr->establish_global_memory_barrier(
			pipeline_stage::acceleration_structure_build,       /* -> */ pipeline_stage::ray_tracing_shaders,
			memory_access::acceleration_structure_write_access, /* -> */ memory_access::acceleration_structure_read_access
		);
		cmdBfr->establish_global_memory_barrier(pipeline_stage::all_commands, pipeline_stage::all_commands,	memory_access::any_write_access, memory_access::any_read_access);

		// GRAPHICS

		switch(mRenderingMethod) {
		case 3: // "Fluid"

			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelineInstanced2->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(mGraphicsPipelineInstanced2);
			cmdBfr->bind_descriptors(mGraphicsPipelineInstanced2->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_indexed(*mSphereIndexBuffer, 10u, 0u, 0u, 0u, *mSphereVertexBuffer, *mPool->particles().hidden_list().get<pbd::hidden_particles::id::pos_backup>().buffer(), *mPool->particles().hidden_list().get<pbd::hidden_particles::id::radius>().buffer());
			cmdBfr->end_render_pass();

			break;
		case 0: // "Instanced Spheres"

			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelineInstanced->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(mGraphicsPipelineInstanced);
			cmdBfr->bind_descriptors(mGraphicsPipelineInstanced->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_indexed(*mSphereIndexBuffer, mNumParticles, 0u, 0u, 0u, *mSphereVertexBuffer, *mParticlesBuffer[ifi]);
			cmdBfr->end_render_pass();

			break;
		case 1: // "Points"

			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelinePoint->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(mGraphicsPipelinePoint);
			cmdBfr->bind_descriptors(mGraphicsPipelinePoint->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_vertices(*mParticlesBuffer[ifi]);
			cmdBfr->end_render_pass();

			break;
		case 2: // "Ray Tracing"

			cmdBfr->bind_pipeline(mRayTracingPipeline);
			cmdBfr->bind_descriptors(mRayTracingPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				binding(0, 0, mCameraDataBuffer[ifi]),
				binding(1, 0, mParticlesBuffer[ifi]),
#if BLAS_CENTRIC
				binding(1, 1, mAabbsBuffer[ifi]->as_storage_buffer()),
#else
				binding(1, 1, mGeometryInstanceBuffers[ifi]->as_storage_buffer()),
#endif
				binding(2, 0, mOffscreenImageViews[ifi]->as_storage_image()),
				binding(2, 1, mTopLevelAS[ifi])                              
			}));

			// Do it:
			cmdBfr->trace_rays(
				for_each_pixel(mainWnd),
				mRayTracingPipeline->shader_binding_table(),
				using_raygen_group_at_index(0),
				using_miss_group_at_index(0),
				using_hit_group_at_index(0)
			);
			
			// Sync ray tracing with transfer:
			cmdBfr->establish_global_memory_barrier(
				pipeline_stage::ray_tracing_shaders,                      pipeline_stage::transfer,
				memory_access::shader_buffers_and_images_write_access,    memory_access::transfer_read_access
			);
			
			copy_image_to_another(mOffscreenImageViews[ifi]->get_image(), mainWnd->current_backbuffer()->image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(cmdBfr, {}, {}));
			
			// Make sure to properly sync with ImGui manager which comes afterwards (it uses a graphics pipeline):
			cmdBfr->establish_global_memory_barrier(
				pipeline_stage::transfer,                                  pipeline_stage::color_attachment_output,
				memory_access::transfer_write_access,                      memory_access::color_attachment_write_access
			);
			
			break;
		default:
			throw std::runtime_error(fmt::format("Invalid mRenderingMethod[{}]", mRenderingMethod));
		}
		
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
		pbd::gpu_list<16>::cleanup();
	}


private: // v== Member variables ==v

	avk::queue* mQueue;
	avk::descriptor_cache mDescriptorCache;

	gvk::quake_camera mQuakeCam;
	std::vector<avk::buffer> mCameraDataBuffer;

	std::unique_ptr<pool> mPool;

	uint32_t mNumParticles;
	std::vector<avk::buffer> mParticlesBuffer;
#if BLAS_CENTRIC
	std::vector<avk::buffer> mAabbsBuffer;
#endif
	avk::buffer mSphereVertexBuffer;
	avk::buffer mSphereIndexBuffer;

#if BLAS_CENTRIC
	std::vector<avk::bottom_level_acceleration_structure> mBottomLevelAS;
#else
	static_assert(INST_CENTRIC);
	avk::bottom_level_acceleration_structure mSingleBlas;
	std::vector<avk::buffer> mGeometryInstanceBuffers;
#endif
	std::vector<avk::top_level_acceleration_structure> mTopLevelAS;

	//avk::compute_pipeline mComputePipeline;
	avk::graphics_pipeline mGraphicsPipelineInstanced;
	avk::graphics_pipeline mGraphicsPipelineInstanced2;
	avk::graphics_pipeline mGraphicsPipelinePoint;
	avk::ray_tracing_pipeline mRayTracingPipeline;
	
	std::vector<avk::image_view> mOffscreenImageViews;

	//avk::buffer mTest;
	
	// Settings from the UI:
	int mRenderingMethod = 3;
	
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
			[](vk::PhysicalDeviceFeatures& deviceFeatures) {
				deviceFeatures.shaderInt64 = VK_TRUE;
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
