#include <cstdio>
#include <cuda_runtime.h>
#include "raytracer.cuh"


__device__ float vector_distance(Vector3d v1, Vector3d v2){
  return pow(pow(v1(0) - v2(0), 2) + pow(v1(1) - v2(1), 2) + pow(v1(2) - v2(2), 2), 1/2);
}

__device__ float vertex_vector_distance(vertex v1, Vector3d v2){
  return pow(pow(v1.x - v2(0), 2) + pow(v1.y - v2(1), 2) + pow(v1.z - v2(2), 2), 1/2);
}

__device__ Vector3d vertexToVector(vertex v){
  return Vector3d(v.x, v.y, v.z);
}
// For each face, we must check for intersection. We store the closest intersection
__device__ bool intersect(object *obj, ray r, vertex *pointHit, vertex *normalHit){
  bool intersection = false;
  float minDistance = FLT_MAX;
  for(int i = 0; i < obj->faces.size(); i++){
    //Calculate the plane that the vertexes of the face lie on
    Vector3d A = vertexToVector(*(obj->vertexes[obj->faces[i]->v1]));
    Vector3d B = vertexToVector(*(obj->vertexes[obj->faces[i]->v2]));
    Vector3d C = vertexToVector(*(obj->vertexes[obj->faces[i]->v3]));

    Vector3d BA = A - B;
    Vector3d CA = A - C;
    Vector3d normal = BA.cross(CA);

    float d = normal.dot(A);

    // Check that the direction of the camera is not parallel to the plane
    if(normal.dot(r.direction) != 0){
      float t = (d - normal.dot(r.origin)) / normal.dot(r.direction);

      // Calculate the point of intersection with the plane, and then use
      // barycentric coordinates to see if we are in the face itself
      Vector3d q = r.origin + t * r.direction;

      if(((B - A).cross(q - A).dot(normal)) >= 0){
        if(((C - B).cross(q - B).dot(normal)) >= 0){
          if(((A - C).cross(q - C).dot(normal)) >= 0){
            intersection = true;
            // If we do have an intersection that is closer than previous intersections
            // , calculate the interpolated normal so
            // that we can calculate the lighting
            if(vector_distance(q, r.origin) < minDistance){
              minDistance = vector_distance(q, r.origin);
              pointHit->x = q(0);
              pointHit->y = q(1);
              pointHit->z = q(2);

              float bary_constant = ((B - A).cross(C - A)).dot(normal);
              float alpha = ((C - B).cross(q - B)).dot(normal) / bary_constant;
              float beta = ((A - C).cross(q - C)).dot(normal) / bary_constant;
              float gamma = ((B - A).cross(q - A)).dot(normal) / bary_constant;

              vertex n1 = *(obj->normal_vertexes[obj->faces[i]->n1]);
              vertex n2 = *(obj->normal_vertexes[obj->faces[i]->n2]);
              vertex n3 = *(obj->normal_vertexes[obj->faces[i]->n3]);

              Vector3d point1 = alpha * vertexToVector(n1);
              Vector3d point2 = beta * vertexToVector(n2);
              Vector3d point3 = gamma * vertexToVector(n3);

              Vector3d interpolated_normal = point1 + point2 + point3;
              interpolated_normal.normalize();

              normalHit->x = interpolated_normal(0);
              normalHit->y = interpolated_normal(1);
              normalHit->z = interpolated_normal(2);
            }
          }
        }
      }
    }
  }
  return intersection;
}

__device__ Vector3d component_wise_product(Vector3d a, Vector3d b){
  float comp1 = a(0) * b(0);
  float comp2 = a(1) * b(1);
  float comp3 = a(2) * b(2);

  Vector3d vec(comp1, comp2, comp3);

  return vec;
}

//Calculates lighting
__device__ color lighting(vertex p, vertex normal, material m, vector<light*> lights,
  vertex c){
  Vector3d diffuse(m.diffuse.r, m.diffuse.g, m.diffuse.b);
  Vector3d ambient(m.ambient.r, m.ambient.g, m.ambient.b);
  Vector3d specular(m.specular.r, m.specular.g, m.specular.b);
  float shine = m.shine.p;

  Vector3d diffuse_sum(0, 0, 0);
  Vector3d specular_sum(0, 0, 0);

  Vector3d position(p.x, p.y, p.z);
  Vector3d camera_pos(c.x, c.y, c.z);

  Vector3d n(normal.x, normal.y, normal.z);

  Vector3d direction = camera_pos - position;
  direction.normalize();

  for(int i = 0; i < lights.size(); i++){
    Vector3d light_pos(lights[i]->x, lights[i]->y ,lights[i]->z);
    Vector3d light_color(lights[i]->r, lights[i]->g, lights[i]->b);
    Vector3d light_direction(light_pos - position);
    float distance = sqrt(light_direction(0) * light_direction(0) +
    light_direction(1) * light_direction(1) +
    light_direction(2) * light_direction(2));
    float attenuation = 1 / (1 + lights[i]->k * pow(distance, 2));
    light_color *= attenuation;
    light_direction.normalize();

    Vector3d light_diffuse;
    if(n.dot(light_direction) < 0){
      light_diffuse << 0, 0, 0;
    }
    else{
      light_diffuse = light_color * n.dot(light_direction);
    }
    diffuse_sum += light_diffuse;

    Vector3d light_specular;

    Vector3d normalized_direction = direction + light_direction;
    normalized_direction.normalize();

    if(n.dot(normalized_direction) < 0){
      light_specular << 0, 0, 0;
    }

    else{
      light_specular = light_color * pow(n.dot(normalized_direction), shine);
    }
    specular_sum += light_specular;
  }


  Vector3d col = ambient + component_wise_product(diffuse_sum, diffuse) +
  component_wise_product(specular_sum, specular);
  if(col(0) > 1){
    col(0) = 1;
  }
  if(col(1) > 1){
    col(1) = 1;
  }
  if(col(2) > 1){
    col(2) = 1;
  }

  color final_color;
  final_color.r = col(0) * 255;
  final_color.g = col(1) * 255;
  final_color.b = col(2) * 255;

  return final_color;
}

__global__
void cudaRaytraceKernel(const object *objects, const light *lights, const camera *c,
color *pixels, const Vector3d e1, const Vector3d e2, const Vector3d e3, float xres, float yres){
  uint idx = threadIdx.x + blockIdx.x * blockDim.x;

  // While we are still dealing with pixels inside of our block
  while(idx < xres * yres){
    // Calculate the direction of the ray (origin is always position of camera)
    // The original direction has been presumed to be (0, 0, -1), and this
    // has been adjusted by the camera orientation outside of the kernel
    ray r;
    r.origin(0) = camera_pos(0);
    r.origin(1) = camera_pos(1);
    r.origin(2) = camera_pos(2);
    int curr_x = idx % xres;
    int curr_y = idx / xres;

    r.direction = c->near * e1 + (curr_x - xres / 2) * 2 * c->right / xres * e2 + e3 * (-curr_y + yres / 2) * 2 * c->top / yres;
    r.direction.normalize();

    // Store the closest pointHit, as well as the normal hit, and the object hit
    vertex pointHit;
    vertex normalHit;
    object *closestObj = NULL;

    // For every object, we will attempt see if the ray and the object intersect.
    // This is done by for every face of every object, we will attempt to see if
    // the ray and the face intersect
    for(int i = 0; i < objects.size(); i++){
      if(intersect(objects[i], r, &pointHit, &normalHit)){
        if(vertex_vector_distance(pointHit, r.origin) < minDistance){
          closestObj = objects[i];
          minDistance = vertex_vector_distance(pointHit, r.origin);
        }
      }
    }

    if(closestObj != NULL){
      vector<light*> lightsAtPixel;
      for(int i = 0; i < lights.size(); i++){
        ray shadowRay;
        bool shadow;
        shadowRay.origin(0) = pointHit.x;
        shadowRay.origin(1) = pointHit.y;
        shadowRay.origin(2) = pointHit.z;

        shadowRay.direction(0) = lights[i]->x - pointHit.x;
        shadowRay.direction(1) = lights[i]->y - pointHit.y;
        shadowRay.direction(2) = lights[i]->z - pointHit.z;

        shadowRay.direction.normalize();
        vertex point;
        vertex normal;
        for(int j = 0; j < objects.size(); j++){
          if(intersect(objects[j], shadowRay, &point, &normal)){
            if(vector_distance(vertexToVector(point), vertexToVector(pointHit)) < .1){
              shadow = true;
            }
            break;
          }
        }
        if(!shadow){
          lightsAtPixel.push_back(lights[i]);
        }
      }
    }
    idx += blockDim.x * gridDim.x;
  }
}

void cudaCallRaytraceKernel(const int blocks, const int threadsPerBlock, const object *objects, const light *lights, const camera *c,
color *pixels, const Vector3d e1, const Vector3d e2, const Vector3d e3, float xres, float yres){
  cudaRaytraceKernel<<<blocks, threadsPerBlock>>>cudaRaytraceKernel(objects, lights, c, pixels,
    e1, e2, e3, xres, yres);
}
