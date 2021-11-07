uint lastNeighborPairIdx;
uint neighborCount = 0u;

void neighborPairInit() {
	if (bool(gpuSettings.mNeighborListSorted)) {
		lastNeighborPairIdx = gl_GlobalInvocationID.x;
		outNeighbors[lastNeighborPairIdx].x = 0u;
	}
}

void addNeighborPair(uint selfId, uint neighborId) {
	uint i = atomicAdd(inOutNeighborsLength, 1u);
	if (i < outNeighbors.length()) {
		if (bool(gpuSettings.mNeighborListSorted)) {
			outNeighbors[gl_GlobalInvocationID.x].x = ++neighborCount;
			outNeighbors[lastNeighborPairIdx    ].y = i;
			outNeighbors[i                      ]   = uvec2(neighborId, 0u);
			lastNeighborPairIdx = i;
		} else {
			outNeighbors[i] = uvec2(selfId, neighborId);
		}
	} else {
		inOutNeighborsLength = outNeighbors.length();
	}
}
