#include "settings.h"
#include <imgui.h>

int   pbd::settings::heightKernelId                = 1;
int   pbd::settings::gradientKernelId              = 1;
bool  pbd::settings::matchGradientToHeightKernel   = true;
bool  pbd::settings::merge                         = false;
bool  pbd::settings::split                         = false;
bool  pbd::settings::baseKernelWidthOnTargetRadius = false;
bool  pbd::settings::updateTargetRadius            = true;
bool  pbd::settings::updateBoundariness            = true;
bool  pbd::settings::groundTruthBoundaryDistance   = true;
float pbd::settings::boundarinessAdaptionSpeed     = 0.01f;
float pbd::settings::kernelWidthAdaptionSpeed      = 0.01f;
int   pbd::settings::color                         = 0;
int   pbd::settings::solverIterations              = 3;

void pbd::settings::add_apbf_settings_im_gui_entries()
{
	ImGui::SliderInt("Solver Iterations", &pbd::settings::solverIterations, 0, 10);
	static const char* const   sHeightKernels[] = { "Cubic", "Gauss", "Poly6", "Cone", "Quadratic Spike" };
	static const char* const sGradientKernels[] = { "Cubic", "Gauss", "Spiky", "Cone", "Quadratic Spike" };
	ImGui::Combo(  "Kernel", &pbd::settings::heightKernelId  , sHeightKernels  , IM_ARRAYSIZE(sHeightKernels)  );
	ImGui::Combo("Gradient", &pbd::settings::gradientKernelId, sGradientKernels, IM_ARRAYSIZE(sGradientKernels));
	ImGui::Checkbox("Match Gradient to Kernel", &pbd::settings::matchGradientToHeightKernel);
	ImGui::Checkbox("Split", &pbd::settings::split);
	ImGui::Checkbox("Merge", &pbd::settings::merge);
	ImGui::Checkbox("Update Boundariness", &pbd::settings::updateBoundariness);
	ImGui::Checkbox("Update Target Radius", &pbd::settings::updateTargetRadius);
	ImGui::Checkbox("Base Kernel Width on Target Radius", &pbd::settings::baseKernelWidthOnTargetRadius);
	ImGui::Checkbox("Ground Truth for Boundary Distance", &pbd::settings::groundTruthBoundaryDistance);
	ImGui::SliderFloat("Boundariness Adaption Speed", &pbd::settings::boundarinessAdaptionSpeed, 0.0f, 1.0f, "%.3f", 2.0f);
	ImGui::SliderFloat("Kernel Width Adaption Speed", &pbd::settings::kernelWidthAdaptionSpeed , 0.0f, 1.0f, "%.3f", 2.0f);
}

void pbd::settings::update_apbf_settings_buffer()
{
	if (matchGradientToHeightKernel) gradientKernelId = heightKernelId;

	apbf_settings apbfSettings;
	apbfSettings.mHeightKernelId                = heightKernelId;
	apbfSettings.mGradientKernelId              = gradientKernelId;
	apbfSettings.mMerge                         = merge;
	apbfSettings.mSplit                         = split;
	apbfSettings.mBaseKernelWidthOnTargetRadius = baseKernelWidthOnTargetRadius;
	apbfSettings.mUpdateTargetRadius            = updateTargetRadius;
	apbfSettings.mUpdateBoundariness            = updateBoundariness;
	apbfSettings.mGroundTruthBoundaryDistance   = groundTruthBoundaryDistance;
	apbfSettings.mBoundarinessAdaptionSpeed     = boundarinessAdaptionSpeed;
	apbfSettings.mKernelWidthAdaptionSpeed      = kernelWidthAdaptionSpeed;
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
