// lattice
#define WIDTH 512 
#define WIDTH4 (WIDTH*4)
#define DEPTH 512
#define DEPTH4 (DEPTH*4)
#define DIRECTIONS 5
#define SIZE (WIDTH*DEPTH*DIRECTIONS)
#define SCALE (2.0f)
#define UNIT (SCALE/((float)WIDTH))
#define LLX (SCALE*-0.5f)
#define LLZ (SCALE*0.5f)

#define TWO_S_D 2.0f*(float)(SCALE)/(float)(DEPTH)
#define TWO_S_W 2.0f*(float)(SCALE)/(float)(WIDTH)
#define D_W (float)(WIDTH)/(float)(DEPTH)

// opencl
#define LWS 8

// display
#define VERTS_NO WIDTH*DEPTH
#define VERTS_NO4 WIDTH*DEPTH*4
#define RENDER_STEPS 2

// convenience
#define store(x, z, d) ((x)*(DEPTH*DIRECTIONS) + (z)*DIRECTIONS + (d))
#define from(x, z, d) from[store((x), (z), (d))]
#define dist(x, z, d) dist[store((x), (z), (d))]
#define omega(m, n, w) omega[(m)*DIRECTIONS + (n)][(w)]
#define buff(x, z) ((x)*WIDTH4 + (z)*4)

// shading
#define FRENEL 0.2

