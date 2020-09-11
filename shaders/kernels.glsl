#define PI 3.14159265

float poly6_kernel_height(vec3 r, float h)
{
	float rSquared = dot(r, r);
	float hSquared = h * h;
	if (rSquared > hSquared) return 0.0f;
	return 315.0f * pow(hSquared - rSquared, 3.0f) / (64.0f * PI * pow(h, 9.0f));
}

vec3 spiky_kernel_gradient(vec3 r, float h)
{
	float dist = length(r);
	if (dist > h || dist < 0.0001f) return vec3(0, 0, 0);
	return -45.0f * pow(h - dist, 2.0f) / (PI * pow(h, 6.0f)) * (r / dist);
}

float cubic_kernel_height(vec3 r, float h)
{
	float dist = length(r);
	if (dist > h) return 0.0f;
	float q = dist / h;
	float k = 8.0f / (PI * h * h * h);
	if (q <= 0.5f)
	{
		float q2 = q * q;
		float q3 = q * q2;
		return k * (6.0f * q3 - 6.0f * q2 + 1.0f);
	}
	return k * (2.0f * pow(1.0f - q, 3.0f));
}

vec3 cubic_kernel_gradient(vec3 r, float h)
{
	float dist = length(r);
	if (dist > h || dist < 0.0001f) return vec3(0, 0, 0);
	float q = dist / h;
	vec3 gradq = r * (1.0f / (dist * h));
	float l = 48.0f / (PI * h * h * h);
	if (q <= 0.5f)
	{
		return l * q * (3.0f * q - 2.0f) * gradq;
	}
	float factor = 1.0f - q;
	return l * (-factor * factor) * gradq;
}

float cone_kernel_height(vec3 r, float h)
{
	float height = 3.0f / (PI * pow(h, DIMENSIONS));
	return max(0.0f, (1.0f - length(r) / h) * height);
}

vec3 cone_kernel_gradient(vec3 r, float h)
{
	float steepness = 3.0f / (PI * pow(h, DIMENSIONS + 1));
	float dist = length(r);
	if (dist > h || dist < 0.0001f) return vec3(0, 0, 0);
	return -r * (steepness / dist);
}

float quadratic_spike_kernel_height(vec3 r, float h)
{
#if DIMENSIONS == 3
	float a = 15.0f / (2 * PI * pow(h, 5));
#else
	float a = 6.0f / (PI * pow(h, 4));
#endif
	return a * pow(min(0, length(r) - h), 2);
}

vec3 quadratic_spike_kernel_gradient(vec3 r, float h)
{
	float dist = length(r);
	if (dist > h || dist < 0.0001f) return vec3(0, 0, 0);
#if DIMENSIONS == 3
	float a = 15.0f / (2 * PI * pow(h, 5));
#else
	float a = 6.0f / (PI * pow(h, 4));
#endif
	return (-2 * a * max(0, h - dist) / dist) * r;
}

float gauss_kernel_height(vec3 r, float height)
{
	float invDoubleVarianceWithoutPi = pow(height, 2.0f / DIMENSIONS);
	float invDoubleVariance = invDoubleVarianceWithoutPi * PI;
	return exp(-dot(r, r) * invDoubleVariance) * pow(invDoubleVarianceWithoutPi, DIMENSIONS / 2.0f);
}

vec3 gauss_kernel_gradient(vec3 r, float height)
{
	float dist = length(r);
	if (dist < 0.0001f) return vec3(0, 0, 0);
	float invDoubleVariance = pow(height, 2.0f / DIMENSIONS) * PI;
	return -gauss_kernel_height(r, height) * 2.0f * dist * invDoubleVariance * normalize(r);
}

float kernel_height(vec3 r, float aKernelWidth)
{
	switch (apbfSettings.mHeightKernelId)
	{
		case 0: return cubic_kernel_height(r, aKernelWidth);
		case 1: return gauss_kernel_height(r, 0.6f / pow(aKernelWidth / 2.0f, DIMENSIONS));
		case 2: return poly6_kernel_height(r, aKernelWidth);
		case 3: return cone_kernel_height(r, aKernelWidth);
		case 4: return quadratic_spike_kernel_height(r, aKernelWidth);
		default: return 0.0f;
	}
}

vec3 kernel_gradient(vec3 r, float aKernelWidth)
{
	switch (apbfSettings.mGradientKernelId)
	{
		case 0: return cubic_kernel_gradient(r, aKernelWidth);
		case 1: return gauss_kernel_gradient(r, 0.6f / pow(aKernelWidth / 2.0f, DIMENSIONS));
		case 2: return spiky_kernel_gradient(r, aKernelWidth);
		case 3: return cone_kernel_gradient(r, aKernelWidth);
		case 4: return quadratic_spike_kernel_gradient(r, aKernelWidth);
		default: return vec3(0.0f, 0.0f, 0.0f);
	}
}
