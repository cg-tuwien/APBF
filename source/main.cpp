#include <gvk.hpp>
#include <imgui.h>
#include <random>
#include "shader_provider.h"
#include "pool.h"
#include "measurements.h"
#include "settings.h"
#include "../shaders/cpu_gpu_shared_config.h"

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
		/** Camera's view matrix */
		glm::mat4 mViewMatrix;
		/** Camera's projection matrix */
		glm::mat4 mProjMatrix;
		/** [0]: time since start, [1]: delta time, [2]: reset particle positions, [3]: set uniform particle radius  */
		glm::vec4 mTimeAndUserInput;
		/** [0]: cullMask for traceRayEXT, [1]: neighborhood-origin particle-id, [2]: perform sphere intersection, [3]: unused  */
		glm::uvec4 mUserInput;
	};

	struct aligned_aabb
	{
		glm::vec3 mMinBounds;
		glm::vec3 mMaxBounds;
		glm::vec2 _align;
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
		using namespace avk;
		using namespace gvk;
		shader_provider::set_updater(&mUpdater.emplace());

#ifdef _DEBUG
		pbd::test::test_all();
#endif
		mPool = std::make_unique<pool>(glm::vec3(-40, -10, -80), glm::vec3(40, 30, -40), 1.0f);
		auto* mainWnd = context().main_window();
		const auto framesInFlight = mainWnd->number_of_frames_in_flight();
		
		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = context().create_descriptor_cache();

		// Create the camera and buffers that will contain camera data:
		mQuakeCam.set_translation({ 0.0f, 0.0f, 0.0f });
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), mainWnd->aspect_ratio(), 0.5f, 500.0f);
		mQuakeCam.disable();
		current_composition()->add_element(mQuakeCam);
		for (window::frame_id_t i = 0; i < framesInFlight; ++i) {
			mCameraDataBuffer.emplace_back(context().create_buffer(
				memory_usage::host_coherent, {},
				uniform_buffer_meta::create_from_data(application_data{})
			));
		}

		for (window::frame_id_t i = 0; i < framesInFlight; ++i) {
			auto imColor     = context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,          vk::Format::eR16G16B16A16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_color_attachment | avk::image_usage::shader_storage);
			auto imNormal    = context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,          vk::Format::eR16G16B16A16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_color_attachment);
			auto imDepth     = context().create_image(mainWnd->resolution().x, mainWnd->resolution().y, format_from_window_depth_buffer(mainWnd), 1, avk::memory_usage::device, avk::image_usage::general_depth_stencil_attachment | avk::image_usage::input_attachment);
			auto imOcclusion = context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,                   vk::Format::eR16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_color_attachment);
			auto imResult    = context().create_image(mainWnd->resolution().x, mainWnd->resolution().y,               vk::Format::eR8G8B8A8Unorm, 1, avk::memory_usage::device, avk::image_usage::general_storage_image);
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
			mImages.emplace_back(context().create_image_view(avk::owned(imColor)), context().create_image_view(avk::owned(imNormal)), context().create_image_view(avk::owned(imDepth)), context().create_image_view(avk::owned(imOcclusion)), context().create_image_view(avk::owned(imResult)));
		}

#if NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT
		
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
			auto scl = glm::vec3{p.mCurrentPositionRadius.w * float{RTX_NEIGHBORHOOD_RADIUS_FACTOR}};
			geomInstInitData.push_back(
				context().create_geometry_instance(mSingleBlas)
				.set_transform_column_major(to_array(glm::translate(glm::mat4{1.0f}, pos) * glm::scale(scl)))
					.set_custom_index(customIndex++)
					.set_mask(NOT_NEIGHBOR_MASK)
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
				geometry_instance_buffer_meta::create_from_data(geomInstInitDataForGpu),
				vertex_buffer_meta::create_from_data(geomInstInitDataForGpu)  .describe_member(sizeof(VkTransformMatrixKHR), vk::Format::eR32Sint, content_description::user_defined_01),
				instance_buffer_meta::create_from_data(geomInstInitDataForGpu).describe_member(sizeof(VkTransformMatrixKHR), vk::Format::eR32Sint, content_description::user_defined_01)
			));
			gib->fill(geomInstInitDataForGpu.data(), 1, sync::wait_idle());
			
			auto& tlas = mTopLevelAS.emplace_back(context().create_top_level_acceleration_structure(mNumParticles, true)); // One top level instance per particle
			tlas->build(gib);
#endif
		}

#endif
		
		// Load a sphere model for drawing a single particle:
		auto sphere = model_t::load_from_file("assets/icosahedron.obj");
		std::tie(mSphereVertexBuffer, mSphereIndexBuffer) = create_vertex_and_index_buffers( make_models_and_meshes_selection(sphere, 0) );

#if NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT
		
		// Create a graphics pipeline for drawing the particles that uses instanced rendering:
		mGraphicsPipelineInstanced = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/instanced.vert"), 
			fragment_shader("shaders/color.frag"),
			// Declare the vertex input to the shaders:
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>()                                                         -> to_location(0), // Declare that positions shall be read from the attached vertex buffer at binding 0, and that we are going to access it in shaders via layout (location = 0)
			from_buffer_binding(1) -> stream_per_instance(&particle::mCurrentPositionRadius)                                 -> to_location(1), // Stream instance data from the buffer at binding 1
			from_buffer_binding(2) -> stream_per_instance(mGeometryInstanceBuffers[0], content_description::user_defined_01) -> to_location(2), // Stream the mask from GeometryInstance data
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
			descriptor_binding(0, 0, mCameraDataBuffer[0])
		);

		//std::vector<glm::vec4> four = { glm::vec4{0, 0, 0, 0}, glm::vec4{0, 3, 0, 0}, glm::vec4{3, 0, 0, 0}, glm::vec4{0, 0, 3, 0} };
		//mTest = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(four));
		//mTest->fill(four.data(), 0, sync::wait_idle());
		
		// Create a graphics pipeline for drawing the particles that uses point primitives:
		mGraphicsPipelinePoint = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/point.vert"), 
			fragment_shader("shaders/color.frag"),
			cfg::polygon_drawing::config_for_points(), cfg::rasterizer_geometry_mode::rasterize_geometry, cfg::culling_mode::disabled,
			// Declare the vertex input to the shaders:
			from_buffer_binding(0) -> stream_per_vertex(&particle::mCurrentPositionRadius)                                 -> to_location(0),
			from_buffer_binding(1) -> stream_per_vertex(mGeometryInstanceBuffers[0], content_description::user_defined_01) -> to_location(1), // Stream the mask from GeometryInstance data
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
			descriptor_binding(0, 0, mCameraDataBuffer[0])
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
			descriptor_binding(0, 0, mCameraDataBuffer[0]),
			descriptor_binding(1, 0, mParticlesBuffer[0]),
#if BLAS_CENTRIC
			descriptor_binding(1, 1, mAabbsBuffer[0]->as_storage_buffer()),
#else 
			descriptor_binding(1, 1, mGeometryInstanceBuffers[0]->as_storage_buffer()),
#endif
			descriptor_binding(2, 0, mOffscreenImageViews[0]->as_storage_image()), // Just take any, this is just to define the layout
			descriptor_binding(3, 0, mTopLevelAS[0])                               // Just take any, this is just to define the layout
		);
#endif
		
		// Get hold of the "ImGui Manager" and add a callback that draws UI elements:
		auto imguiManager = current_composition()->element_by_type<imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([this](){
				ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
				ImGui::Text("%.3f ms/Simulation Step", measurements::get_timing_interval_in_ms("PBD"));
				ImGui::Text("%.3f ms/Neighborhood", measurements::get_timing_interval_in_ms("Neighborhood"));
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
#if NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT
				static const char* const sRenderingMethods[] = {"Instanced Spheres", "Points", "Ray Tracing", "Fluid"};
				ImGui::Combo("Rendering Method", &mRenderingMethod, sRenderingMethods, IM_ARRAYSIZE(sRenderingMethods));
				static const char* const sNeighborRenderOptions[] = {"All", "Only Neighbors", "All But Neighbors"};
				ImGui::Combo("Particles to Render", &mRenderNeighbors, sNeighborRenderOptions, IM_ARRAYSIZE(sNeighborRenderOptions));
				ImGui::DragInt("Neighborhood-Origin Particle-ID", &mNeighborhoodOriginParticleId, 1, 0, static_cast<int>(mNumParticles));
#endif
				ImGui::Checkbox("Freeze Particle Animation", &mFreezeParticleAnimation);
#if NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT
				ImGui::Checkbox("Reset Particle Positions", &mResetParticlePositions);
				ImGui::Checkbox("Set Uniform Particle Radius", &mSetUniformParticleRadius);
				static const char* const sIntersectionTypes[] = {"AABB Intersection", "Sphere Intersection"};
				ImGui::Combo("Neighborhood Intersection", &mIntersectionType, sIntersectionTypes, IM_ARRAYSIZE(sIntersectionTypes));
#endif
				ImGui::Checkbox("Render Boxes", &mPool->mRenderBoxes);
				ImGui::Checkbox("Ambient Occlusion", &mAddAmbientOcclusion);
				static const char* const sColors[] = { "Boundariness", "Boundary Distance", "Transferring", "Kernel Width", "Target Radius", "Radius", "Velocity" };
				ImGui::Combo("Color", &pbd::settings::color, sColors, IM_ARRAYSIZE(sColors));
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
		using namespace gvk;

		if (input().key_pressed(key_code::f1) || input().mouse_button_pressed(2) || input().mouse_button_released(2)) {
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

		if (input().key_pressed(key_code::space)) {
			mFreezeParticleAnimation = !mFreezeParticleAnimation;
		}

		if (input().key_pressed(key_code::enter)) {
			mPerformSingleSimulationStep = true;
		}

		if (input().key_pressed(key_code::backspace)) {
			mPool->time_machine().jump_back();
		}

		if (input().key_pressed(key_code::t)) {
			mPool->time_machine().toggle_enabled();
		}
		
		// On Esc pressed,
		if (input().key_pressed(key_code::escape)) {
			// stop the current composition:
			current_composition()->stop();
		}

		if (!mQuakeCam.is_enabled() && !mImGuiHovered) {
			mPool->handle_input(glm::inverse(mQuakeCam.projection_and_view_matrix()), mQuakeCam.translation());
		}
	}

	void render() override
	{
		using namespace avk;
		using namespace gvk;

		auto* mainWnd = context().main_window();
		const auto ifi = mainWnd->current_in_flight_index();

		static float sTimeSinceStartForAnimation = 0.0f;
		if (!mFreezeParticleAnimation) {
			sTimeSinceStartForAnimation = time().time_since_start();
		}
		application_data cd{
			// Camera's view matrix
			mQuakeCam.view_matrix(),
			// Camera's projection matrix
			mQuakeCam.projection_matrix(),
			// [0]: time since start, [1]: delta time, [2]: reset particle positions, [3]: set uniform particle radius 
			glm::vec4 {
				sTimeSinceStartForAnimation,
				time().delta_time(),
				mResetParticlePositions ? 1.0f : 0.0f,
				mSetUniformParticleRadius ? 1.0f : 0.0f
			},
			// [0]: cullMask for traceRayEXT, [1]: neighborhood-origin particle-id, [2]: perform sphere intersection, [3]: unused
			glm::uvec4{
				0 == mRenderNeighbors ? /* all: */ uint32_t{0xFF} : 1 == mRenderNeighbors ? /* neighbors: */ uint32_t{NEIGHBOR_MASK} : /* not neighbors: */ uint32_t{NOT_NEIGHBOR_MASK},
				static_cast<uint32_t>(mNeighborhoodOriginParticleId),
				mIntersectionType,
				0u
			}
		};
		mCameraDataBuffer[ifi]->fill(&cd, 0, sync::not_required());
		pbd::settings::update_apbf_settings_buffer();

		shader_provider::start_recording();
		measurements::record_timing_interval_start("PBD");

		if (!mFreezeParticleAnimation || mPerformSingleSimulationStep) {
			mPool->update(std::min(gvk::time().delta_time(), mMaxDeltaTime));
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
		auto color1 = glm::vec3(0, 0.2, 0);
		auto color2 = glm::vec3(1, 0.2, 0);
		auto color1Float = 0.0f;
		auto color2Float = 1.0f;
		auto isUint = false;
		auto isParticleProperty = false;
		switch (pbd::settings::color) {
			case 0: floatForColor = mPool->fluid().get<pbd::fluid::id::boundariness     >();                                                    break;
			case 1: floatForColor = mPool->fluid().get<pbd::fluid::id::boundary_distance>();        color2Float = POS_RESOLUTION *  20; isUint = true; break;
			case 2: floatForColor = transferring;                        isParticleProperty = true; color1Float = 0; color2Float =   1; isUint = true; break;
			case 3: floatForColor = mPool->fluid().get<pbd::fluid::id::kernel_width     >();        color1Float = 4; color2Float =   8;                break;
			case 4: floatForColor = mPool->fluid().get<pbd::fluid::id::target_radius    >();        color1Float = 1; color2Float =   2;                break;
			case 5: floatForColor = radius;                                                         color1Float = 1; color2Float = 1.3f;               break;
			case 6: floatForColor = mPool->scalar_particle_velocities(); isParticleProperty = true; color1Float = 0; color2Float =  10;                break;
		}
		if (isParticleProperty) {
			auto old = floatForColor;
			shader_provider::copy_scattered_read(old.buffer(), floatForColor.write().buffer(), idx.index_buffer(), idx.length(), 4);
			floatForColor.set_length(mPool->fluid().length());
		}
		if (isUint) {
			shader_provider::uint_to_float(floatForColor.write().buffer(), floatForColor.write().buffer(), floatForColor.write().length(), 1.0f);
		}

		measurements::record_timing_interval_end("PBD");

#if NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT
		
		// COMPUTE

#if BLAS_CENTRIC
		shader_provider::roundandround(mCameraDataBuffer[ifi], mParticlesBuffer[ifi], mAabbsBuffer[ifi], mTopLevelAS[ifi], mNumParticles);
#else
		shader_provider::roundandround(mCameraDataBuffer[ifi], mParticlesBuffer[ifi], mGeometryInstanceBuffers[ifi], mTopLevelAS[ifi], mNumParticles);
		shader_provider::cmd_bfr()->establish_global_memory_barrier(
			pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::compute_shader,
			memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::shader_buffers_and_images_any_access
		);
		shader_provider::mask_neighborhood(mCameraDataBuffer[ifi], mParticlesBuffer[ifi], mGeometryInstanceBuffers[ifi], mTopLevelAS[ifi], mNumParticles);
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
		
#else
		auto& cmdBfr = shader_provider::cmd_bfr();
#endif

		// GRAPHICS

		measurements::debug_label_start("Rendering", glm::vec4(0, 0.5, 0, 1));

		switch (mRenderingMethod) {
		case 3: // "Fluid"
		{
			auto fragToVS = glm::inverse(mQuakeCam.projection_matrix()) * glm::translate(glm::vec3(-1, -1, 0)) * glm::scale(glm::vec3(2.0f / glm::vec2(mainWnd->resolution()), 1.0f));
			auto result = &mImages[ifi].mColor;
			static auto lengthLimit = pbd::gpu_list<4>().request_length(1); // TODO maybe more elegant solution? Or just remove this debug functionality
			if (pbd::settings::particleRenderLimit != 0) lengthLimit.set_length(pbd::settings::particleRenderLimit);
			auto& particleCount = pbd::settings::particleRenderLimit == 0 ? position.length() : lengthLimit.length();

			shader_provider::render_particles(mCameraDataBuffer[ifi], mSphereVertexBuffer, mSphereIndexBuffer, position.buffer(), radius.buffer(), floatForColor.buffer(), particleCount, mImages[ifi].mNormal, mImages[ifi].mDepth, mImages[ifi].mColor, static_cast<uint32_t>(mSphereIndexBuffer->meta_at_index<generic_buffer_meta>().num_elements()), color1, color2, color1Float, color2Float, pbd::settings::particleRenderScale);
			if (mAddAmbientOcclusion) {
				shader_provider::render_ambient_occlusion(mCameraDataBuffer[ifi], mSphereVertexBuffer, mSphereIndexBuffer, position.buffer(), radius.buffer(), position.length(), mImages[ifi].mNormal, mImages[ifi].mDepth, mImages[ifi].mOcclusion, static_cast<uint32_t>(mSphereIndexBuffer->meta_at_index<generic_buffer_meta>().num_elements()), fragToVS, pbd::settings::particleRenderScale);
				shader_provider::darken_image(mImages[ifi].mOcclusion, mImages[ifi].mColor, mImages[ifi].mResult, 0.7f);
				result = &mImages[ifi].mResult;
			}
			blit_image           (          (*result)->get_image(), mainWnd->current_backbuffer()->image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));
			copy_image_to_another(mImages[ifi].mDepth->get_image(), mainWnd->current_backbuffer()->image_view_at(1)->get_image(), sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));
			shader_provider::sync_after_transfer();

			break;
		}
		case 0: // "Instanced Spheres"

			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelineInstanced->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(avk::const_referenced(mGraphicsPipelineInstanced));
			cmdBfr->bind_descriptors(mGraphicsPipelineInstanced->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_indexed(avk::const_referenced(mSphereIndexBuffer), mNumParticles, 0u, 0u, 0u, avk::const_referenced(mSphereVertexBuffer), avk::const_referenced(mParticlesBuffer[ifi]), avk::const_referenced(mGeometryInstanceBuffers[ifi]));
			cmdBfr->end_render_pass();

			break;
		case 1: // "Points"

			cmdBfr->begin_render_pass_for_framebuffer(mGraphicsPipelinePoint->get_renderpass(), mainWnd->current_backbuffer());
			cmdBfr->bind_pipeline(avk::const_referenced(mGraphicsPipelinePoint));
			cmdBfr->bind_descriptors(mGraphicsPipelinePoint->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mCameraDataBuffer[ifi])
			}));
			cmdBfr->draw_vertices(avk::const_referenced(mParticlesBuffer[ifi]), avk::const_referenced(mGeometryInstanceBuffers[ifi]));
			cmdBfr->end_render_pass();

			break;
		case 2: // "Ray Tracing"

			cmdBfr->bind_pipeline(avk::const_referenced(mRayTracingPipeline));
			cmdBfr->bind_descriptors(mRayTracingPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mCameraDataBuffer[ifi]),
				descriptor_binding(1, 0, mParticlesBuffer[ifi]),
#if BLAS_CENTRIC
				descriptor_binding(1, 1, mAabbsBuffer[ifi]->as_storage_buffer()),
#else
				descriptor_binding(1, 1, mGeometryInstanceBuffers[ifi]->as_storage_buffer()),
#endif
				descriptor_binding(2, 0, mOffscreenImageViews[ifi]->as_storage_image()),
				descriptor_binding(3, 0, mTopLevelAS[ifi])
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

			copy_image_to_another(mOffscreenImageViews[ifi]->get_image(), mainWnd->current_backbuffer()->image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(*cmdBfr, {}, {}));

			// Make sure to properly sync with ImGui manager which comes afterwards (it uses a graphics pipeline):
			cmdBfr->establish_global_memory_barrier(
				pipeline_stage::transfer,                                  pipeline_stage::color_attachment_output,
				memory_access::transfer_write_access,                      memory_access::color_attachment_write_access
			);

			break;
		default:
			throw std::runtime_error(fmt::format("Invalid mRenderingMethod[{}]", mRenderingMethod));
		}
		mPool->render(mQuakeCam.projection_and_view_matrix()); // TODO won't work if NEIGHBORHOOD_RTX_PROOF_OF_CONCEPT is defined

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
	std::vector<images> mImages;

	bool mImGuiHovered = false;
	
	// Settings from the UI:
	int mRenderingMethod = 3;
	int mRenderNeighbors = 1;
	int mNeighborhoodOriginParticleId = 0;
	bool mFreezeParticleAnimation = true;
	bool mAddAmbientOcclusion = true;
	bool mResetParticlePositions = false;
	bool mSetUniformParticleRadius = false;
	bool mPerformSingleSimulationStep = false;
	int mIntersectionType = 0;
	float mMaxDeltaTime = 0.1f;
	
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
		mainWnd->set_presentaton_mode(presentation_mode::fifo);
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
