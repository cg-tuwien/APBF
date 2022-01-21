#include "incompressibility.h"
#include "measurements.h"
#include "settings.h"

pbd::incompressibility& pbd::incompressibility::set_data(fluid* aFluid, neighbors* aNeighbors)
{
	mFluid = aFluid;
	mNeighbors = aNeighbors;
	return *this;
}

void pbd::incompressibility::apply()
{
	measurements::debug_label_start("Incompressibility constraints");

	auto& kernelWidthList      = mFluid->get<fluid::id::kernel_width>();
	auto& boundarinessList     = mFluid->get<fluid::id::boundariness>();
	auto& particleList         = mFluid->get<fluid::id::particle>();
	auto& positionList         = particleList.hidden_list().get<pbd::hidden_particles::id::position>();
	auto& inverseMassList      = particleList.hidden_list().get<pbd::hidden_particles::id::inverse_mass>();
	auto& radiusList           = particleList.hidden_list().get<pbd::hidden_particles::id::radius>();
	auto  incompDataList       = pbd::gpu_list<32>().request_length(mFluid->requested_length());
	auto  scaledGradientList   = pbd::gpu_list<16>().request_length(mNeighbors->requested_length());
	auto  lambdaList           = pbd::gpu_list< 4>().request_length(mFluid->requested_length());
	auto  centerOfMassList     = pbd::gpu_list<16>().request_length(mFluid->requested_length());

	// optimization: centerOfMassList only needed for one boundariness calc method; avoid write() allocation and just pass some other buffer
	auto centerOfMassBuffer = &kernelWidthList.buffer();
	if (settings::boundarinessCalculationMethod == 1) {
		centerOfMassBuffer = &centerOfMassList.write().buffer();
		shader_provider::write_sequence(*centerOfMassBuffer, particleList.length(), 0u, 0u, 4u);
	}

	// incompressibility_0: initialization of incompDataList, contribution of every particle to its own density
	// incompressibility_1: contribution of neighbors to the density and the gradient; optional center of mass stuff
	// incompressibility_2: use density and gradient to compute lambda and boundariness, begin to solve the constraint by moving the "center particle" (at the center of the kernel)
	// incompressibility_3: finish solving the constraint by moving neighbors

	measurements::debug_label_start("0"); shader_provider::incompressibility_0(particleList.index_buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), incompDataList.write().buffer(), particleList.length());                                                                                                                                  measurements::debug_label_end();
	measurements::debug_label_start("1"); shader_provider::incompressibility_1(particleList.index_buffer(), positionList.buffer(), radiusList.buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), mNeighbors->buffer(), incompDataList.write().buffer(), *centerOfMassBuffer, scaledGradientList.write().buffer(), mNeighbors->length());       measurements::debug_label_end();
	measurements::debug_label_start("2"); shader_provider::incompressibility_2(particleList.index_buffer(), positionList.write().buffer(), radiusList.buffer(), inverseMassList.buffer(), kernelWidthList.buffer(), incompDataList.buffer(), *centerOfMassBuffer, boundarinessList.write().buffer(), lambdaList.write().buffer(), particleList.length()); measurements::debug_label_end();
	measurements::debug_label_start("3"); shader_provider::incompressibility_3(particleList.index_buffer(), inverseMassList.buffer(), mNeighbors->buffer(), scaledGradientList.buffer(), lambdaList.buffer(), incompDataList.buffer(), positionList.write().buffer(), boundarinessList.write().buffer(), mNeighbors->length());                           measurements::debug_label_end();

	measurements::debug_label_end();
}
