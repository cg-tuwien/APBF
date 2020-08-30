#include "settings.h"
#include <imgui.h>

int  pbd::settings::kernelId                      = 1;
bool pbd::settings::merge                         = false;
bool pbd::settings::split                         = false;
bool pbd::settings::baseKernelWidthOnTargetRadius = false;
bool pbd::settings::updateTargetRadius            = true;
bool pbd::settings::updateBoundariness            = true;
int  pbd::settings::color                         = 0;

void pbd::settings::add_apbf_settings_im_gui_entries()
{
	ImGui::Checkbox("Split", &pbd::settings::split);
	ImGui::Checkbox("Merge", &pbd::settings::merge);
	ImGui::Checkbox("Update Boundariness", &pbd::settings::updateBoundariness);
	ImGui::Checkbox("Update Target Radius", &pbd::settings::updateTargetRadius);
	ImGui::Checkbox("Base Kernel Width on Target Radius", &pbd::settings::baseKernelWidthOnTargetRadius);
}

void pbd::settings::update_apbf_settings_buffer()
{
	apbf_settings apbfSettings;
	apbfSettings.mKernelId                      = kernelId;
	apbfSettings.mMerge                         = merge;
	apbfSettings.mSplit                         = split;
	apbfSettings.mBaseKernelWidthOnTargetRadius = baseKernelWidthOnTargetRadius;
	apbfSettings.mUpdateTargetRadius            = updateTargetRadius;
	apbfSettings.mUpdateBoundariness            = updateBoundariness;
	apbf_settings_buffer()->fill(&apbfSettings, 0, avk::sync::not_required());
}

avk::buffer& pbd::settings::apbf_settings_buffer()
{
	static avk::buffer result = gvk::context().create_buffer(
		avk::memory_usage::host_coherent, {},
		avk::uniform_buffer_meta::create_from_data(apbf_settings{})
	);
	return result;
}
