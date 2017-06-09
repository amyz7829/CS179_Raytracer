#include <cstdio>
#include <cuda_runtime.h>
using namespace Eigen;

float vector_distance(Vector3d v1, Vector3d v2);
float vertex_vector_distance(vertex v1, Vector3d v2);
Vector3d vertexToVector(vertex v);
bool intersect(object *obj, ray r, vertex *pointHit, vertex *normalHit);

Vector3d component_wise_product(Vector3d a, Vector3d b)
color lighting(vertex p, vertex normal, material m, vector<light*> lights,
	vertex c);
void cudaRaytraceKernel(const object *objects, const light *lights, const camera *c,
  color *pixels, const Vector3d e1, const Vector3d e2, const Vector3d e3, float xres, float yres)

void cudaCallRaytraceKernel(const int blocks, const int threadsPerBlock, const object *objects, const light *lights, const camera *c,
  color *pixels, const Vector3d e1, const Vector3d e2, const Vector3d e3, float xres, float yres)
