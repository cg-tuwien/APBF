
#define BLAS_CENTRIC 0
#if !BLAS_CENTRIC
#define INST_CENTRIC 1
#else
#define INST_CENTRIC 0
#endif

#define NOT_NEIGHBOR_MASK  0x01
#define NEIGHBOR_MASK      0x02
#define ORIGIN_MASK        0x06

#define RTX_NEIGHBORHOOD_RADIUS_FACTOR 30

#define UNIFORM_PARTICLE_RADIUS 0.07
