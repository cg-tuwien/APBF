#include "settings.h"
#include <imgui.h>
#include "../shaders/cpu_gpu_shared_config.h"

int   pbd::settings::heightKernelId                    = 1;
int   pbd::settings::gradientKernelId                  = 1;
bool  pbd::settings::matchGradientToHeightKernel       = true;
bool  pbd::settings::merge                             = true;
bool  pbd::settings::split                             = true;
bool  pbd::settings::baseKernelWidthOnTargetRadius     = true;
bool  pbd::settings::baseKernelWidthOnBoundaryDistance = true;
bool  pbd::settings::updateTargetRadius                = true;
bool  pbd::settings::updateBoundariness                = true;
bool  pbd::settings::neighborListSorted                = false;
bool  pbd::settings::groundTruthBoundaryDistance       = false;
bool  pbd::settings::renderBoxes                       = true;
bool  pbd::settings::basicPbf                          = false;
float pbd::settings::boundarinessAdaptionSpeed         = 0.5f;
float pbd::settings::kernelWidthAdaptionSpeed          = 0.01f;
float pbd::settings::boundarinessSelfGradLengthFactor  = 8.0f;
float pbd::settings::boundarinessUnderpressureFactor   = 4.0f;
float pbd::settings::neighborBoundarinessThreshold     = 0.0f;
float pbd::settings::mergeDuration                     = 2.0f;
float pbd::settings::splitDuration                     = 0.0f;
float pbd::settings::smallestTargetRadius              = 1.0f;
float pbd::settings::targetRadiusOffset                = 10.0f;
float pbd::settings::targetRadiusScaleFactor           = DIMENSIONS == 3 ? 0.3f : 0.4f;
float pbd::settings::particleRenderScale               = 0.7f;
int   pbd::settings::particleRenderLimit               = 0;
int   pbd::settings::color                             = 0;
int   pbd::settings::solverIterations                  = 3;

const std::vector<const char*> pbd::settings::colorNames = std::vector<const char*>({ "Boundariness", "Boundary Distance", "Transferring", "Kernel Width", "Target Radius", "Radius", "Velocity" });

void pbd::settings::add_apbf_settings_im_gui_entries()
{
	ImGui::SliderInt("Solver Iterations", &pbd::settings::solverIterations, 0, 10);
	static const char* const   sHeightKernels[] = { "Cubic", "Gauss", "Poly6", "Cone", "Quadratic Spike" };
	static const char* const sGradientKernels[] = { "Cubic", "Gauss", "Spiky", "Cone", "Quadratic Spike" };
	ImGui::Combo(  "Kernel", &pbd::settings::heightKernelId  , sHeightKernels  , IM_ARRAYSIZE(sHeightKernels)  );
	ImGui::Combo("Gradient", &pbd::settings::gradientKernelId, sGradientKernels, IM_ARRAYSIZE(sGradientKernels));
	ImGui::Checkbox("Match Gradient to Kernel", &pbd::settings::matchGradientToHeightKernel);
	ImGui::Checkbox("Basic PBF", &pbd::settings::basicPbf);
	ImGui::Checkbox("Split", &pbd::settings::split);
	ImGui::Checkbox("Merge", &pbd::settings::merge);
	ImGui::SliderFloat("Split Duration", &pbd::settings::splitDuration, 0.0f, 4.0f, "%.1f");
	ImGui::SliderFloat("Merge Duration", &pbd::settings::mergeDuration, 0.0f, 4.0f, "%.1f");
	ImGui::Checkbox("Neighbor List Sorted", &pbd::settings::neighborListSorted);
//	ImGui::Checkbox("Update Boundariness", &pbd::settings::updateBoundariness);
//	ImGui::Checkbox("Update Target Radius", &pbd::settings::updateTargetRadius);
	ImGui::Checkbox("Base Kernel Width on Target Radius", &pbd::settings::baseKernelWidthOnTargetRadius);
	ImGui::Checkbox("Base Kernel Width on Boundary Distance", &pbd::settings::baseKernelWidthOnBoundaryDistance);
//	ImGui::Checkbox("Ground Truth for Boundary Distance", &pbd::settings::groundTruthBoundaryDistance);
	ImGui::SliderFloat("Boundariness Adaption Speed", &pbd::settings::boundarinessAdaptionSpeed, 0.0f, 1.0f, "%.3f", 2.0f);
	ImGui::SliderFloat("Kernel Width Adaption Speed", &pbd::settings::kernelWidthAdaptionSpeed , 0.0f, 1.0f, "%.3f", 2.0f);

	ImGui::SliderFloat("Smallest Target Radius", &pbd::settings::smallestTargetRadius, 0.1f, 8.0f, "%.1f", 2.0f);
	ImGui::SliderFloat("Target Radius Offset", &pbd::settings::targetRadiusOffset, 0.0f, 40.0f, "%.0f");
	ImGui::SliderFloat("Target Radius Scale Factor", &pbd::settings::targetRadiusScaleFactor, 0.0f, 1.0f, "%.2f");

	ImGui::SliderFloat("Self Gradient Length Factor", &pbd::settings::boundarinessSelfGradLengthFactor, 0.0f, 16.0f, "%.1f");
	ImGui::SliderFloat("Underpressure Factor", &pbd::settings::boundarinessUnderpressureFactor, 0.0f, 16.0f, "%.1f");
//	ImGui::SliderFloat("Neighbor Boundariness Threshold", &pbd::settings::neighborBoundarinessThreshold, 0.0f, 1.0f, "%.2f");
}

void pbd::settings::update_apbf_settings_buffer()
{
	if (matchGradientToHeightKernel) gradientKernelId = heightKernelId;

	apbf_settings apbfSettings;
	apbfSettings.mHeightKernelId                    = heightKernelId;
	apbfSettings.mGradientKernelId                  = gradientKernelId;
	apbfSettings.mMerge                             = merge;
	apbfSettings.mSplit                             = split;
	apbfSettings.mBaseKernelWidthOnTargetRadius     = baseKernelWidthOnTargetRadius;
	apbfSettings.mBaseKernelWidthOnBoundaryDistance = baseKernelWidthOnBoundaryDistance;
	apbfSettings.mUpdateTargetRadius                = updateTargetRadius;
	apbfSettings.mUpdateBoundariness                = updateBoundariness;
	apbfSettings.mNeighborListSorted                = neighborListSorted;
	apbfSettings.mGroundTruthBoundaryDistance       = groundTruthBoundaryDistance;
	apbfSettings.mBoundarinessAdaptionSpeed         = boundarinessAdaptionSpeed;
	apbfSettings.mKernelWidthAdaptionSpeed          = kernelWidthAdaptionSpeed;
	apbfSettings.mBoundarinessSelfGradLengthFactor  = boundarinessSelfGradLengthFactor;
	apbfSettings.mBoundarinessUnderpressureFactor   = boundarinessUnderpressureFactor;
	apbfSettings.mNeighborBoundarinessThreshold     = neighborBoundarinessThreshold;
	apbfSettings.mMergeDuration                     = mergeDuration;
	apbfSettings.mSmallestTargetRadius              = smallestTargetRadius;
	apbfSettings.mTargetRadiusOffset                = targetRadiusOffset;
	apbfSettings.mTargetRadiusScaleFactor           = targetRadiusScaleFactor;
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

void pbd::settings::nextColor()
{
	color++;
	if (color >= colorNames.size()) color = 0;
}

void pbd::settings::previousColor()
{
	if (color <= 0) color = static_cast<int>(colorNames.size());
	color--;
}
