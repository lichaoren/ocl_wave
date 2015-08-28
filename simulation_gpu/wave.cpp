/*
 * The program uses 8 wave numbers since the float8, the obstruction is done
 * by routing density reversely to ci. There is no dissipation because it
 * requires extra procedure. The surface is toroidal. The whole program is
 * limited in 500 lines.
 *
 *  Created on: Nov 15, 2014
 *      Author: chaorel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <GL/glew.h>
#include <GL/glfw.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>

#include "RGU.h"
#include "wave.h"

// OCL vars
size_t gws[2] = {WIDTH, DEPTH};
size_t lws[2] = {LWS, LWS};
cl_command_queue myqueue;
cl_context mycontext;
cl_kernel mykrn_update, mykrn_heights, mykrn_normals, mykrn_colors;
cl_mem f_ocl[2], dist_ocl, omega_ocl, rbuf_ocl, nbuf_ocl;
cl_int err_ocl;

// OGL vars
const GLuint vertex = 0;
const GLuint indx = 1;
const GLuint indx_size = (WIDTH-1)*(DEPTH-1)*4*sizeof(GLuint);
GLuint buf_ogl[2];
GLuint indices[(WIDTH-1)*(DEPTH-1)][4];
GLfloat rbuf[2*VERTS_NO4];
float mat_ambient[] = {0.0, 0.0, 0.0, 1.0};
float mat_diffuse[] = {0.79, 0.63, 0.29, 1.0};
float mat_specular[] = {1.0, 1.0, 1.0, 1.0};
float mat_shininess[] = {100.0};
float eye_pos[4] = {-2.0, 2.0, 2.0, 0.0}; // warning: [3] must be 0
float light_dir[4] = {1.0, 1.0, 1.0, 0.0}; // warning: [3] must be 0

// data vars
#define WAVENUMBERS 8
const float wavenos[] = {0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15};
int ci[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
int invci[] = {0, 2, 1, 4, 3};
int half = 48;
float cubewidth = half*UNIT*2.f-0.05f; // shrink a little to show collision

// host vars
bool mask[VERTS_NO];
int dist[SIZE];
float f[2][SIZE][WAVENUMBERS];
float omega[DIRECTIONS*DIRECTIONS][WAVENUMBERS];

// convenience
float dot4(float a[4], float b[4]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

void cross4(float dest[4], float a[4], float b[4]) {
  dest[0] = a[1]*b[2]-a[2]*b[1];
  dest[1] = -a[0]*b[2]+a[2]*b[0];
  dest[2] = a[0]*b[1]-a[1]*b[0];
  dest[3] = 0.0;
}

void normalize4(float dest[4]) {
  float len = sqrt(dot4(dest, dest));
  dest[0] /= len; dest[1] /= len; dest[2] /= len; dest[3] /= len;
}

int magnitude(int x, int z) {
  return sqrt(x*x + z*z);
}

void init_lattice() {
  int x, z, d, w;
  float height;

  for (x = 0; x<WIDTH; ++x) {
    for (z = 0; z<DEPTH; ++z) {
      for (d = 0; d<DIRECTIONS; d++) {
        for (w = 0; w<WAVENUMBERS; ++w) {
          f[0][store(x, z, d)][w] = 0.0;
          f[1][store(x, z, d)][w] = 0.0;
        }
        height =  magnitude(x-384, z-128);
        if (height < half) {
          height /= 1000;
          for (w = 0; w<WAVENUMBERS; ++w)
            f[0][store(x, z, d)][w] = height/(float)(WAVENUMBERS);
          f[0][store(x, z, 1)][w] *= mask[x*DEPTH+z];
        }
      }
    }
  }
}

void init_dist() {
  int x, z, d, tx, tz;
  int xlow = WIDTH/2-half;
  int xhigh = WIDTH/2+half;
  int zlow = DEPTH/2-half;
  int zhigh = DEPTH/2+half;

  for (x = 0; x<WIDTH; x++) {
    for (z = 0; z<DEPTH; z++) {
      for (d = 0; d<DIRECTIONS; d++) {
        tx = (x+ci[d][0]+WIDTH)%WIDTH;
        tz = (z+ci[d][1]+DEPTH)%DEPTH;
        dist[store(x, z, d)] = store(tx, tz, d);

        if (x>=xlow&&x<=xhigh && z>=zlow&&z<=zhigh) {
          mask[x*DEPTH+z] = 0;
          dist[store(x, z, d)] = store(x, z, invci[d]);
        }
        else mask[x*DEPTH+z] = 1;
      }
    }
  }
}

void init_omega() {
  int x, z, w;

  for (w = 0; w<WAVENUMBERS; ++w) {
    omega(0, 0, w)= -4.0*wavenos[w];
    for (z = 1; z<DIRECTIONS; z++) omega(0, z, w)= 2.0-4.0*wavenos[w];
    for (x = 1; x<DIRECTIONS; x++) omega(x, 0, w)= wavenos[w];
    for (x = 1; x<3; x++) {
      for (z = 1; z<3; z++) omega(x, z, w)= wavenos[w]-1.0;
      for (z=3;z<5;z++) omega(x, z, w) = wavenos[w];
    }
    for (x = 3; x<5; x++) {
      for (z = 1; z<3; z++) omega(x, z, w)= wavenos[w];
      for(z=3;z<5;z++) omega(x, z, w) = wavenos[w]-1.0;
    }
  }
}

void load_data() {
  init_dist();
  init_lattice();
  init_omega();
}

void load_vertices() {
  int x, z;

  for (x = 0; x<WIDTH; ++x) {
    for (z = 0; z<DEPTH; ++z) {
      rbuf[x*DEPTH4+z*4+0] = LLX+float(z)*UNIT;
      rbuf[x*DEPTH4+z*4+1] = 0.0f;
      rbuf[x*DEPTH4+z*4+2] = LLZ-float(x)*UNIT;
      rbuf[x*DEPTH4+z*4+3] = 1.0f;
      rbuf[VERTS_NO4+x*DEPTH4+z*4+0] = 0.0f;
      rbuf[VERTS_NO4+x*DEPTH4+z*4+1] = 1.0f;
      rbuf[VERTS_NO4+x*DEPTH4+z*4+2] = 0.0f;
      rbuf[VERTS_NO4+x*DEPTH4+z*4+3] = 1.0f;
    }
  }
}

void load_indices() {
  int x, z;
  for (x = 0; x<WIDTH-1; x++) {
    for (z = 0; z<DEPTH-1; z++) {
      indices[x*(DEPTH-1)+z][0] = x*DEPTH+z;
      indices[x*(DEPTH-1)+z][1] = (x+1)*DEPTH+z;
      indices[x*(DEPTH-1)+z][2] = (x+1)*DEPTH+1+z;
      indices[x*(DEPTH-1)+z][3] = x*DEPTH+1+z;
    }
  }
}

void set_lights() {
  float light0_ambient[] = {0.1, 0.1, 0.1, 1.0};
  float light0_diffuse[] = {1.0, 1.0, 1.0, 1.0};
  float light0_specular[] = {1.0, 1.0, 1.0, 1.0};
  float light0_position[] = {light_dir[0], light_dir[1], light_dir[2],
      light_dir[3]};

  glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
}

void do_materials()
{
  glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
  glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
}

void set_camera() {
  float eye[] = { eye_pos[0], eye_pos[1], eye_pos[2]};
  float view[] = { 0.0, 0.0, 0.0};
  float up[]  = { 0.0, 1.0, 0.0 };

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, 1.0, 0.1, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(eye[0], eye[1], eye[2],
            view[0], view[1], view[2],
            up[0], up[1], up[2]);
}

void initGL(int argc, char** argv)
{
  load_indices();
  load_vertices();

  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE);
  glutInitWindowSize(768,768);
  glutInitWindowPosition(100,50);
  glutCreateWindow("my_cool_cube");
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_SMOOTH);
  glClearColor(0.1, 0.1, 0.1, 1.0);
  glewInit();

  set_camera();
  set_lights();
  glGenBuffers(2, buf_ogl);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf_ogl[indx]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indx_size,
      (GLvoid*)(&indices[0][0]), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, buf_ogl[vertex]);
  glBufferData(GL_ARRAY_BUFFER, VERTS_NO4*2*sizeof(GLfloat),
		  (GLvoid*)(&rbuf[0]), GL_DYNAMIC_DRAW);
  glVertexPointer(4, GL_FLOAT, 0, (GLvoid*)(0));
  glColorPointer(4, GL_FLOAT, 0, (GLvoid*)(VERTS_NO4*sizeof(GLfloat)));
  return;
}

void bufferCL() {
  f_ocl[0]=clCreateBuffer(mycontext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
    SIZE*sizeof(cl_float8), &f[0][0][0], &err_ocl);
  f_ocl[1]=clCreateBuffer(mycontext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
    SIZE*sizeof(cl_float8), &f[1][0][0], &err_ocl);
  dist_ocl=clCreateBuffer(mycontext, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
    SIZE*sizeof(int), &dist[0], &err_ocl);
  omega_ocl=clCreateBuffer(mycontext, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
    DIRECTIONS*DIRECTIONS*sizeof(cl_float8), &(omega[0][0]), &err_ocl);
  rbuf_ocl=clCreateFromGLBuffer(mycontext, CL_MEM_READ_WRITE, buf_ogl[vertex],
    &err_ocl);
  nbuf_ocl = clCreateBuffer(mycontext, CL_MEM_READ_WRITE,
      WIDTH*DEPTH*sizeof(cl_float4), NULL, &err_ocl);

  clSetKernelArg(mykrn_update, 2, sizeof(cl_mem), (void *) &dist_ocl);
  clSetKernelArg(mykrn_update, 3, sizeof(cl_mem), (void *) &omega_ocl);
  clSetKernelArg(mykrn_heights, 0, sizeof(cl_mem), (void *) &rbuf_ocl);
  clSetKernelArg(mykrn_normals, 0, sizeof(cl_mem), (void *) &rbuf_ocl);
  clSetKernelArg(mykrn_normals, 1, sizeof(cl_mem), (void *) &nbuf_ocl);
  clSetKernelArg(mykrn_colors, 0, sizeof(cl_mem), (void *) &rbuf_ocl);
  clSetKernelArg(mykrn_colors, 1, sizeof(cl_mem), (void *) &nbuf_ocl);
  float L[4] = {light_dir[0], light_dir[1], light_dir[2], light_dir[3]};
  normalize4(L);
  clSetKernelArg(mykrn_colors, 2, sizeof(cl_float4), &L);
  clSetKernelArg(mykrn_colors, 3, sizeof(cl_float4), &eye_pos);
}

void initCL() {
  cl_platform_id myplatform;
  cl_device_id * mydevice;
  cl_program myprogram;
  size_t prog_len;
  unsigned int gpudevcount;
  const char* kernelHeader;
  char* kernelSource;

  err_ocl=RGUGetPlatformID(&myplatform);
  err_ocl=clGetDeviceIDs(myplatform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpudevcount);
  mydevice=new cl_device_id[gpudevcount];
  err_ocl=clGetDeviceIDs(myplatform, CL_DEVICE_TYPE_GPU, gpudevcount, mydevice,
    NULL);
  cl_context_properties props[]= {
  CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
  CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
  CL_CONTEXT_PLATFORM, (cl_context_properties) myplatform, 0 };
  mycontext=clCreateContext(props, 1, &mydevice[0], NULL, NULL, &err_ocl);
  myqueue=clCreateCommandQueue(mycontext, mydevice[0], 0, &err_ocl);
  kernelHeader=RGULoadProgSource("wave.h", "", &prog_len);
  kernelSource=RGULoadProgSource("wave.cl", kernelHeader, &prog_len);
  myprogram=clCreateProgramWithSource(mycontext, 1,
    (const char**) &kernelSource, &prog_len, &err_ocl);
  err_ocl=clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
  if (err_ocl==CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(myprogram, mydevice[0],
    CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log=(char *) malloc(log_size);
    clGetProgramBuildInfo(myprogram, mydevice[0],
    CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("error when building program\n %s\n", log);
  }
  mykrn_update=clCreateKernel(myprogram, "update", &err_ocl);
  mykrn_heights=clCreateKernel(myprogram, "heights", &err_ocl);
  mykrn_normals = clCreateKernel(myprogram, "normals", &err_ocl);
  mykrn_colors = clCreateKernel(myprogram, "colors", &err_ocl);

  free(kernelSource);
  clReleaseProgram(myprogram);

  bufferCL();
}

void run_updates() {
  static int from = 0;
  static int t = 0;
  cl_event wait[1];

  clSetKernelArg(mykrn_update, 0, sizeof(cl_mem), (void *) &f_ocl[from]);
  clSetKernelArg(mykrn_update, 1, sizeof(cl_mem), (void *) &f_ocl[1-from]);
  clEnqueueNDRangeKernel(myqueue, mykrn_update, 2,
      NULL, gws, lws, 0, 0, &wait[0]);
  clWaitForEvents(1, wait);
  if (t%RENDER_STEPS==0) {
    clSetKernelArg(mykrn_heights, 1, sizeof(cl_mem), (void *) &f_ocl[1-from]);
    clEnqueueNDRangeKernel(myqueue, mykrn_heights, 2,
        NULL, gws, lws, 0, 0, &wait[0]);
    clWaitForEvents(1, wait);
    clEnqueueNDRangeKernel(myqueue, mykrn_normals, 2, NULL, gws, lws, 0, 0,
        &wait[0]);
    clWaitForEvents(1, wait);
    clEnqueueNDRangeKernel(myqueue, mykrn_colors, 2, NULL, gws, lws, 0, 0,
        &wait[0]);
    clWaitForEvents(1, wait);
  }

  from = 1-from;
  ++t;
  usleep(10000);
}

void drawShape()
{
  glEnable(GL_LIGHTING);
  do_materials();
  glPushMatrix();
  glScaled(1, 2, 1);
  glutSolidCube(cubewidth);
  glPopMatrix();
  glPushMatrix();
  glScaled(1, 2, 1);
  glRotatef(45, 0, 1, 0);
  glRotatef(90, 1, 0, 0);
  glutSolidTorus(0.05, 1.5, 5, 4);
  glPopMatrix();
  glDisable(GL_LIGHTING);
}

void mydisplayfunc()
{
  glFinish();
  clEnqueueAcquireGLObjects(myqueue, 1, &rbuf_ocl, 0, 0, 0);
  run_updates();
  clEnqueueReleaseGLObjects(myqueue, 1, &rbuf_ocl, 0, 0, 0);
  clFinish(myqueue);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawShape();
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawElements(GL_QUADS, (WIDTH-1)*(DEPTH-1)*4, GL_UNSIGNED_INT,
      (GLvoid*)(0));
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glutSwapBuffers();
  glutPostRedisplay();
}

void cleanup()
{
  clReleaseMemObject(f_ocl[0]);
  clReleaseMemObject(f_ocl[1]);
  clReleaseMemObject(dist_ocl);
  clReleaseMemObject(omega_ocl);
  clReleaseMemObject(rbuf_ocl);
  clReleaseMemObject(nbuf_ocl);
  clReleaseCommandQueue(myqueue);
  clReleaseContext(mycontext);
  clReleaseKernel(mykrn_update);
  clReleaseKernel(mykrn_heights);
  clReleaseKernel(mykrn_normals);
  clReleaseKernel(mykrn_colors);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(2,buf_ogl);
  exit(EXIT_SUCCESS);
}

void getout(unsigned char key, int x, int y) {
  switch (key) {
    case 'q':
      cleanup();
    break;
    default:
    break;
  }
}

int main(int argc, char** argv)
{
  load_data();
  initGL(argc, argv);
  initCL();
  glutDisplayFunc(mydisplayfunc);
  glutKeyboardFunc(getout);
  glutMainLoop();
  return 0;
}
