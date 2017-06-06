#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cstring>
#include <string>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "../Eigen/Dense"
#include <sstream>
#include <map>

using namespace std;
using namespace Eigen;

// The vertex structure contains 3 coordinates.
struct vertex{
	float x;
	float y;
	float z;
};

struct color{
	float r;
	float g;
	float b;
};

// The face structure contains 3 vertexes, which are one-indexed
struct face{
 	int v1;
 	int v2;
 	int v3;

 	int n1;
 	int n2;
 	int n3;
};

struct coordinate{
	float x;
	float y;
	bool valid;
};

struct light{
	float x;
	float y;
	float z;

	float r;
	float g;
	float b;

	float k;
};

struct shininess{
	float p;
};

struct material{
	color ambient;
	color diffuse;
	color specular;
	shininess shine;
};

struct colored_vertex{
	vertex v;
	color col;
};

// The object structure contains the vertexes, faces, and transformation matrices
struct object{
	string file_name;
	string name;
	material m;
	vector <vertex *> vertexes;
	vector <vertex *> normal_vertexes;
	vector <coordinate *> coordinates;
	vector <face *> faces;
	vector <Matrix4d> transformations;
	vector <Matrix4d> normal_transformations;
};

struct orientation{
	float x;
	float y;
	float z;
	float angle;
};

struct camera{
	vertex position;
	orientation o;
	float near;
	float far;
	float left;
	float right;
	float top;
	float bottom;
};


struct ray{
	Vector3d origin;
	Vector3d direction;
};

int parse_obj_file(const char* obj_name, object *obj);
void parse_camera_data(ifstream *parser, camera *c);
vector<light*> parse_light_data(ifstream *parser);
string find_material_reflectances(ifstream *parser, object *obj);
int find_transformations(ifstream *parser, object *obj, string buf);
Matrix4d create_camera_matrix(camera *c);
Matrix4d create_perspective_matrix(camera *c);
Vector3d component_wise_product(Vector3d a, Vector3d b);
color lighting(vertex p, vertex normal, material m, vector<light*> lights, vertex c);
Matrix4d makeTranslationMatrix(vertex v);
Matrix4d makeRotationMatrix(orientation o);
float vector_distance(Vector3d v1, Vector3d v2);
float vertex_vector_distance(vertex v1, Vector3d v2);
Vector3d vertexToVector(vertex v);
bool intersect(object *obj, ray r, vertex *pointHit, vertex *normalHit);
void raytrace(vector<object*> objects, vector<light*> lights, camera *c, color *pixels, int xres, int yres);
