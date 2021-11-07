#include "settings.h"
#include <imgui.h>
#include "../shaders/cpu_gpu_shared_config.h"

bool  pbd::settings::bruteForceActive                   = true;
bool  pbd::settings::rtxActive                          = true;
bool  pbd::settings::greenActive                        = false; // Green implementation contains bugs and is slow, so disable it by default
bool  pbd::settings::binarySearchActive                 = true;
bool  pbd::settings::showFocusParticleNeighborhoodRange = false;
bool  pbd::settings::neighborListSorted                 = false;
float pbd::settings::particleRenderScale                = 0.1f;
int   pbd::settings::focusParticleId                    = 0;
int   pbd::settings::renderedNeighborhoodMethod         = 0;

const std::vector<const char*> pbd::settings::neighborhoodMethodNames = std::vector<const char*>({ "Brute Force", "RTX", "Green", "Binary Search" });

void pbd::settings::add_settings_im_gui_entries()
{
	ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Active:");
	ImGui::Checkbox("Brute Force"  , &pbd::settings::bruteForceActive  );
	ImGui::Checkbox("RTX"          , &pbd::settings::rtxActive         );
	ImGui::Checkbox("Green"        , &pbd::settings::greenActive       );
	ImGui::Checkbox("Binary Search", &pbd::settings::binarySearchActive);

	ImGui::Separator();

	ImGui::Combo("Rendered Method", &pbd::settings::renderedNeighborhoodMethod, pbd::settings::neighborhoodMethodNames.data(), static_cast<int>(pbd::settings::neighborhoodMethodNames.size()));
	ImGui::SliderFloat("Render Scale", &pbd::settings::particleRenderScale, 0.0f, 1.0f, "%.1f");
	ImGui::SliderInt("Focus Particle", &pbd::settings::focusParticleId, 0, PARTICLE_COUNT - 1);
	ImGui::Checkbox("Show its range", &pbd::settings::showFocusParticleNeighborhoodRange);
	ImGui::Checkbox("Neighbor List Sorted", &pbd::settings::neighborListSorted);
}

void pbd::settings::update_settings_buffer()
{
	gpu_settings gpuSettings;
	gpuSettings.mParticleRenderScale                = particleRenderScale;
	gpuSettings.mNeighborListSorted                 = neighborListSorted;
	gpuSettings.mFocusParticleId                    = focusParticleId;
	gpuSettings.mShowFocusParticleNeighborhoodRange = showFocusParticleNeighborhoodRange;
	settings_buffer()->fill(&gpuSettings, 0, avk::sync::not_required());
}

avk::buffer& pbd::settings::settings_buffer()
{
	static avk::buffer result = gvk::context().create_buffer(
		avk::memory_usage::host_coherent, {},
		avk::uniform_buffer_meta::create_from_data(gpu_settings{})
	);
	return result;
}
