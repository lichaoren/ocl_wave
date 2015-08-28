#define FREQUENCY 4.0f

__kernel void update(__global float8* from, __global float8* to,
          __global int* dist, __global float8* omega)
{
  unsigned int x = get_global_id(0);
  unsigned int z = get_global_id(1);
  unsigned  int m, n;
  __private unsigned int dist_index[DIRECTIONS];
  __private float8 value[DIRECTIONS];
  __private float8 compo;

  for( m = 0;m < DIRECTIONS; m++) {
    dist_index[m] = dist(x,z,m);
    value[m] = to[dist_index[m]] = from(x, z, m);
  }

  for( m = 0; m < DIRECTIONS; m++ )  {
    compo = (float8)(0.0);
    for( n = 0; n < DIRECTIONS; n++)
       compo += omega[DIRECTIONS*m + n]*value[n];
    to[dist_index[m]] += compo;
  }
}

__kernel void heights(__global float* rbuf,__global float8* from)
{
  unsigned int x = get_global_id(0);
  unsigned int z = get_global_id(1);

  float sum = 0;
  unsigned int d;
  for(d = 0; d < DIRECTIONS; ++d)
  {
    float8 compo = from(x, z, d);
    sum += compo.s0 + compo.s1 + compo.s2 + compo.s3
         + compo.s4 + compo.s5 + compo.s6 + compo.s7;
  }

  rbuf[x*DEPTH4 + z*4 + 1] = sum;
}

__kernel void normals(__global float* rbuf, __global float4* nbuf)
{
  unsigned int x = get_global_id(0);
  unsigned int z = get_global_id(1);

  unsigned int y1 = buff(x==(WIDTH-1) ? (WIDTH-1) : x+1, z) + 1;
  unsigned int y2 = buff(x==0 ? 0 : x-1, z) + 1;
  unsigned int y3 = buff(x, z==(DEPTH-1) ? (WIDTH-1) : z+1) + 1;
  unsigned int y4 = buff(x, z==0 ? 0: z-1) + 1;

  float4 norm = TWO_S_D * (float4)(
    rbuf[y2]-rbuf[y1], TWO_S_W, D_W*(rbuf[y4]-rbuf[y3]), 0.0f);

  nbuf[x*DEPTH+z] = fast_normalize(norm);
}

__kernel void colors(__global float4* rbuf, __global float4* nbuf,
                     float4 L, float4 eye)
{
  unsigned int x = get_global_id(0);
  unsigned int z = get_global_id(1);
  float4 ligh = (float4)(0.2, 0.6, 0.8, 0.4);
  float4 dark = (float4)(0.0, 0.15, 0.3, 0.8);
  float4 sky = (float4)(0.2, 0.1, 0.5, 1.0);

  float4 pos = rbuf[x*DEPTH + z];
  float4 V = fast_normalize(pos - eye);
  V.w = 0;

  float seaview = -min(0.0, dot(L, V));
  float4 color = dot(nbuf[x*DEPTH + z], L)
      * (mix(dark, ligh, seaview) + FRENEL*sky*pow((1.0-seaview), 5.0));

  rbuf[VERTS_NO + x*DEPTH + z] = color;
}
