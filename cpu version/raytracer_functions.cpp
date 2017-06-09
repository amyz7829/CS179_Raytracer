#include "raytracer.h"

//Takes pointer to a parser and a camera struct to record the important
//camera data from the file. Should end on an empty line
void parse_camera_data(ifstream *parser, camera *c){
	//Read line
	string buf;
	getline(*parser, buf);

	//Create a stringstream
	stringstream ss;
	ss.clear();
	ss.str("");

	//Until a new line is hit, keep parsing
	while((int)buf[0] != 0){
		//Always clear the string stream first
		ss.clear();
		ss.str("");
		ss<<buf;
		string type;
		if(buf.find("position") != string::npos){
			//Type = position, x, y, z are the vertex coords
			string type;
			float x;
			float y;
			float z;
			ss>>type>>x>>y>>z;

			vertex position;
			position.x = x;
			position.y = y;
			position.z = z;

			c->position = position;
		}
		if(buf.find("orientation") != string::npos){
			orientation o;

			ss>>type>>o.x>>o.y>>o.z>>o.angle;

			c->o = o;
		}
		if(buf.find("near") != string::npos){
			float near;
			ss>>type>>near;

			c->near = near;
		}
		if(buf.find("far") != string::npos){
			float far;
			ss>>type>>far;

			c->far = far;
		}
		if(buf.find("left") != string::npos){
			float left;
			ss>>type>>left;

			c->left = left;
		}
		if(buf.find("right") != string::npos){
			float right;
			ss>>type>>right;

			c->right = right;
		}
		if(buf.find("top") != string::npos){
			float top;
			ss>>type>>top;

			c->top = top;
		}
		if(buf.find("bottom") != string::npos){
			float bottom;
			ss>>type>>bottom;

			c->bottom = bottom;
		}

		//Move forward another line
		getline(*parser, buf);
	}
}

//Should start with an empty line, and then first line it calls getline on
//is a light line
vector<light*> parse_light_data(ifstream *parser){
	//Read line
	string buf;
	getline(*parser, buf);
	vector<light*> lights;

	stringstream ss;
	ss.clear();
	ss.str("");

	while((int)buf[0] != 0){
		ss.clear();
		ss.str("");
		ss<<buf;
		string type;
		float x, y, z, r, g, b, k;
		string sep;

		ss>>type>>x>>y>>z>>sep>>r>>g>>b>>sep>>k;

		light *l = new light;
		l->x = x;
		l->y = y;
		l->z = z;

		l->r = r;
		l->g = g;
		l->b = b;

		l->k = k;
		lights.push_back(l);

		getline(*parser, buf);
	}
	return lights;
}


//Parse through an object file and get the faces and vertexes
int parse_obj_file(const char* obj_name, object *obj){
	// The ifstream object that parses the files
	ifstream parser;
	parser.open(obj_name);

	// Check that the file has been opened properly
	if(!parser.good()){
		cerr << "Something has gone wrong when opening your file. Please try"<<
		"again." << endl;
		return 1;
	}
	else{
		// Create a vector of vertex pointers that is one-indexed
		vector<vertex *> vertexes;
		vertexes.push_back(NULL);

		vector<vertex *> normal_vertexes;
		normal_vertexes.push_back(NULL);

		// Create a vector of face pointers
		vector<face *> faces;
		vector<face *> normal_faces;

		while(!parser.eof()){
			// Read in a line
			string buf;
			getline(parser, buf);
			stringstream ss;
			ss.clear();
			ss.str("");
			// Check that the line starts with a valid character
			if(buf[0] == 'v' || buf[0] == 'f' || (int)buf[0] == 0){
				ss<<buf;
				if(buf[0] == 'v'){
					string type;
					float x;
					float y;
					float z;

					//Get the variables from the stringstream
					ss>>type>>x>>y>>z;

				    // Sets the x, y, z points for each vertex
				    vertex *v = new vertex;
				    v -> x = x;
				    v -> y = y;
				    v -> z = z;

					if(buf[1] == 'n'){
						normal_vertexes.push_back(v);
					}
					else{
					    vertexes.push_back(v);
					}
				}
				if(buf[0] == 'f'){
					string type;
					string v1;
					string v2;
					string v3;

					//Get the variables from the stringstream
					ss>>type>>v1>>v2>>v3;

					int fv1, fv2, fv3;
					int fnv1, fnv2, fnv3;

					ss.clear();
					ss.str(v1);

					string item;
					getline(ss, item, '/');
					fv1 = atoi(item.c_str());


					getline(ss, item, '/');
					getline(ss, item, '/');
					fnv1 = atoi(item.c_str());

					ss.clear();
					ss.str(v2);

					getline(ss, item, '/');
					fv2 = atoi(item.c_str());

					getline(ss, item, '/');
					getline(ss, item, '/');
					fnv2 = atoi(item.c_str());

					ss.clear();
					ss.str(v3);

					getline(ss, item, '/');
					fv3 = atoi(item.c_str());

					getline(ss, item, '/');
					getline(ss, item, '/');
					fnv3 = atoi(item.c_str());

				    // Sets the vertexes for each face
				    face *f = new face;
				    f -> v1 = fv1;
				    f -> v2 = fv2;
				    f -> v3 = fv3;

				    f -> n1 = fnv1;
				    f -> n2 = fnv2;
				    f -> n3 = fnv3;


				    faces.push_back(f);
				}
			}
		}
		obj->faces = faces;
		obj->vertexes = vertexes;
		obj->normal_vertexes = normal_vertexes;

		parser.close();
		return 0;
	}
}

string find_material_reflectances(ifstream *parser, object *obj){
	//Read line
	string buf;
	getline(*parser, buf);
	//Create a stringstream
	stringstream ss;
	ss.clear();
	ss.str("");

	//Until we are done looking at ambient/diffuse/specular/shininesss
	while(buf.find("ambient") != string::npos ||
		buf.find("diffuse") != string::npos ||
		buf.find("specular") != string::npos ||
		buf.find("shininess") != string::npos){
		//Always clear the string stream first
		ss.clear();
		ss.str("");
		ss<<buf;
		string type;
		if(buf.find("ambient") != string::npos){
			string type;
			float r;
			float g;
			float b;
			ss>>type>>r>>g>>b;

			color a;
			a.r = r;
			a.g = g;
			a.b = b;

			obj->m.ambient = a;
		}
		if(buf.find("diffuse") != string::npos){
			string type;
			float r;
			float g;
			float b;
			ss>>type>>r>>g>>b;

			color d;
			d.r = r;
			d.g = g;
			d.b = b;

			obj->m.diffuse = d;
		}
		if(buf.find("specular") != string::npos){
			string type;
			float r;
			float g;
			float b;
			ss>>type>>r>>g>>b;

			color s;
			s.r = r;
			s.g = g;
			s.b = b;

			obj->m.specular = s;
		}
		if(buf.find("shininess") != string::npos){
			string type;
			float p;
			ss>>type>>p;

			shininess s;
			s.p = p;

			obj->m.shine = s;
		}
		getline(*parser, buf);
	}
	return buf;
}

//Finds the transformations in the .txt file and attaches them to the obj file.
//Returns 0 if successful and 1 if not
int find_transformations(ifstream *parser, object *obj, string buf){
	// Create a vector to hold all of the transformations
	vector<Matrix4d> transformations;
	vector<Matrix4d> normal_transformations;
	while((int)buf[0] != 0){
		stringstream ss;
		ss.clear();
		ss.str("");
		// Check that the line starts with a valid character
		if(buf[0] == 's' || buf[0] == 't' || buf[0] == 'r'){
			//Load buf into stringstream
			ss<<buf;
			if(buf[0] == 's'){
				string type;
			    float x;
			    float y;
			    float z;

			    //Get the variables from the stringstream
			    ss>>type>>x>>y>>z;

			    //Creates the scaling matrix
			    Matrix4d m(4,4);
			    m << x, 0, 0, 0, //row 1
			    	 0, y, 0, 0, //row 2
			    	 0, 0, z, 0, //row 3
			    	 0, 0, 0, 1; //row 4

			   	transformations.push_back(m);
			   	normal_transformations.push_back(m);
			}
			if(buf[0] == 'r'){
				string type;
			    float x;
			    float y;
			    float z;
			    float a;

			    //Get the variables from the stringstream
			    ss>>type>>x>>y>>z>>a;


			    double mag = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
			    double ux = x / mag;
			    double uy = y / mag;
			    double uz = z / mag;

			 	//Create the rotation matrix
			    Matrix4d m(4, 4);
			    m << pow(ux, 2) + (1 - pow(ux, 2)) * cos(a), ux * uy * (1 - cos(a)) - uz * sin(a), ux * uz * (1 - cos(a)) + uy * sin(a), 0, //row 1
			   		 uy * ux * (1 - cos(a)) + uz * sin(a), pow(uy, 2) + (1 - pow(uy, 2)) * cos(a), uy * uz * (1 - cos(a)) - ux * sin(a), 0, //row 2
			   		 ux * uz * (1 - cos(a)) - uy * sin(a), uy * uz * (1 - cos(a)) + ux * sin(a), pow(uz, 2) + (1 - pow(uz, 2)) * cos(a), 0, //row 3
			   		 0, 								0,									  0,									    1; //row 4

			   	transformations.push_back(m);
			   	normal_transformations.push_back(m);
			}
			if(buf[0] == 't'){
				string type;
			    float x;
			    float y;
			    float z;

			    //Get the variables from the stringstream
			    ss>>type>>x>>y>>z;


			    //Create the translation matrix
			    Matrix4d m(4, 4);
			    m << 1, 0, 0, x, //row 1
			     	 0, 1, 0, y, //row 2
			     	 0, 0, 1, z, //row 3
			     	 0, 0, 0, 1; //row 4

			    transformations.push_back(m);
			}
		}
		// Read in a line
		getline(*parser, buf);
	}

  obj->transformations = transformations;
	obj->normal_transformations = normal_transformations;

  return 0;
}

Matrix4d makeTranslationMatrix(vertex v){
  //Create the translation matrix
  Matrix4d m(4, 4);
  m << 1, 0, 0, v.x, //row 1
     0, 1, 0, v.y, //row 2
     0, 0, 1, v.z, //row 3
     0, 0, 0, 1; //row 4

  return m;
}

Matrix4d makeRotationMatrix(orientation o){
  double mag = sqrt(pow(o.x, 2) + pow(o.y, 2) + pow(o.z, 2));
  double ux = o.x / mag;
  double uy = o.y / mag;
  double uz = o.z / mag;

  //Create the rotation matrix
  Matrix4d m(4, 4);
  m << pow(ux, 2) + (1 - pow(ux, 2)) * cos(o.angle), ux * uy * (1 - cos(o.angle)) - uz * sin(o.angle), ux * uz * (1 - cos(o.angle)) + uy * sin(o.angle), 0, //row 1
     uy * ux * (1 - cos(o.angle)) + uz * sin(o.angle), pow(uy, 2) + (1 - pow(uy, 2)) * cos(o.angle), uy * uz * (1 - cos(o.angle)) - ux * sin(o.angle), 0, //row 2
     ux * uz * (1 - cos(o.angle)) - uy * sin(o.angle), uy * uz * (1 - cos(o.angle)) + ux * sin(o.angle), pow(uz, 2) + (1 - pow(uz, 2)) * cos(o.angle), 0, //row 3
     0, 								0,									  0,									    1; //row 4
  return m;
}


// float vector_distance(Vector3d v1, Vector3d v2){
//   return pow(pow(v1(0) - v2(0), 2) + pow(v1(1) - v2(1), 2) + pow(v1(2) - v2(2), 2), 1/2);
// }
//
// float vertex_vector_distance(vertex v1, Vector3d v2){
//   return pow(pow(v1.x - v2(0), 2) + pow(v1.y - v2(1), 2) + pow(v1.z - v2(2), 2), 1/2);
// }
//
// Vector3d vertexToVector(vertex v){
//   return Vector3d(v.x, v.y, v.z);
// }
// // For each face, we must check for intersection. We store the closest intersection
// bool intersect(object *obj, ray r, vertex *pointHit, vertex *normalHit){
//   bool intersection = false;
//   float minDistance = FLT_MAX;
//   for(int i = 0; i < obj->faces.size(); i++){
//     //Calculate the plane that the vertexes of the face lie on
//     Vector3d A = vertexToVector(*(obj->vertexes[obj->faces[i]->v1]));
//     Vector3d B = vertexToVector(*(obj->vertexes[obj->faces[i]->v2]));
//     Vector3d C = vertexToVector(*(obj->vertexes[obj->faces[i]->v3]));
//
//     Vector3d BA = A - B;
//     Vector3d CA = A - C;
//     Vector3d normal = BA.cross(CA);
//
//     float d = normal.dot(A);
//
//     // Check that the direction of the camera is not parallel to the plane
//     if(normal.dot(r.direction) != 0){
//       float t = (d - normal.dot(r.origin)) / normal.dot(r.direction);
//
//       // Calculate the point of intersection with the plane, and then use
//       // barycentric coordinates to see if we are in the face itself
//       Vector3d q = r.origin + t * r.direction;
//
//       if(((B - A).cross(q - A).dot(normal)) >= 0){
//         if(((C - B).cross(q - B).dot(normal)) >= 0){
//           if(((A - C).cross(q - C).dot(normal)) >= 0){
//             intersection = true;
//             // If we do have an intersection that is closer than previous intersections
//             // , calculate the interpolated normal so
//             // that we can calculate the lighting
//             if(vector_distance(q, r.origin) < minDistance){
//               minDistance = vector_distance(q, r.origin);
//               pointHit->x = q(0);
//               pointHit->y = q(1);
//               pointHit->z = q(2);
//
//               float bary_constant = ((B - A).cross(C - A)).dot(normal);
//               float alpha = ((C - B).cross(q - B)).dot(normal) / bary_constant;
//               float beta = ((A - C).cross(q - C)).dot(normal) / bary_constant;
//               float gamma = ((B - A).cross(q - A)).dot(normal) / bary_constant;
//
//               vertex n1 = *(obj->normal_vertexes[obj->faces[i]->n1]);
//               vertex n2 = *(obj->normal_vertexes[obj->faces[i]->n2]);
//               vertex n3 = *(obj->normal_vertexes[obj->faces[i]->n3]);
//
//               Vector3d point1 = alpha * vertexToVector(n1);
//               Vector3d point2 = beta * vertexToVector(n2);
//               Vector3d point3 = gamma * vertexToVector(n3);
//
//               Vector3d interpolated_normal = point1 + point2 + point3;
//               interpolated_normal.normalize();
//
//               normalHit->x = interpolated_normal(0);
//               normalHit->y = interpolated_normal(1);
//               normalHit->z = interpolated_normal(2);
//             }
//           }
//         }
//       }
//     }
//   }
//   return intersection;
// }
//
// Vector3d component_wise_product(Vector3d a, Vector3d b){
// 	float comp1 = a(0) * b(0);
// 	float comp2 = a(1) * b(1);
// 	float comp3 = a(2) * b(2);
//
// 	Vector3d vec(comp1, comp2, comp3);
//
// 	return vec;
// }
//
// //Calculates lighting
// color lighting(vertex p, vertex normal, material m, vector<light*> lights,
// 	vertex c){
// 	Vector3d diffuse(m.diffuse.r, m.diffuse.g, m.diffuse.b);
// 	Vector3d ambient(m.ambient.r, m.ambient.g, m.ambient.b);
// 	Vector3d specular(m.specular.r, m.specular.g, m.specular.b);
// 	float shine = m.shine.p;
//
// 	Vector3d diffuse_sum(0, 0, 0);
// 	Vector3d specular_sum(0, 0, 0);
//
// 	Vector3d position(p.x, p.y, p.z);
// 	Vector3d camera_pos(c.x, c.y, c.z);
//
// 	Vector3d n(normal.x, normal.y, normal.z);
//
// 	Vector3d direction = camera_pos - position;
// 	direction.normalize();
//
// 	for(int i = 0; i < lights.size(); i++){
// 		Vector3d light_pos(lights[i]->x, lights[i]->y ,lights[i]->z);
// 		Vector3d light_color(lights[i]->r, lights[i]->g, lights[i]->b);
// 		Vector3d light_direction(light_pos - position);
// 		float distance = sqrt(light_direction(0) * light_direction(0) +
// 			light_direction(1) * light_direction(1) +
// 			light_direction(2) * light_direction(2));
// 		float attenuation = 1 / (1 + lights[i]->k * pow(distance, 2));
// 		light_color *= attenuation;
// 		light_direction.normalize();
//
// 		Vector3d light_diffuse;
// 		if(n.dot(light_direction) < 0){
// 			light_diffuse << 0, 0, 0;
// 		}
// 		else{
// 			light_diffuse = light_color * n.dot(light_direction);
// 		}
// 		diffuse_sum += light_diffuse;
//
// 		Vector3d light_specular;
//
// 		Vector3d normalized_direction = direction + light_direction;
// 		normalized_direction.normalize();
//
// 		if(n.dot(normalized_direction) < 0){
// 			light_specular << 0, 0, 0;
// 		}
//
// 		else{
// 			light_specular = light_color * pow(n.dot(normalized_direction), shine);
// 		}
// 		specular_sum += light_specular;
// 	}
//
//
// 	Vector3d col = ambient + component_wise_product(diffuse_sum, diffuse) +
// 	component_wise_product(specular_sum, specular);
// 	if(col(0) > 1){
// 		col(0) = 1;
// 	}
// 	if(col(1) > 1){
// 		col(1) = 1;
// 	}
// 	if(col(2) > 1){
// 		col(2) = 1;
// 	}
//
// 	color final_color;
// 	final_color.r = col(0) * 255;
// 	final_color.g = col(1) * 255;
// 	final_color.b = col(2) * 255;
//
// 	return final_color;
// }

void raytrace(vector<object*> objects, vector<light*> lights, camera *c, color pixels[], int xres, int yres){
  int x = xres;
  int y = yres;

  Vector3d e1;
  Vector3d e2;
  Vector3d e3;

  // Adjusts the camera based on the settings, e1 is the original direction it is facing, transform
  Vector3d camera_pos = vertexToVector(c->position);
  e1(0) = (makeRotationMatrix(c->o) * Vector4d(0, 0, -1, 1))(0);
  e1(1) = (makeRotationMatrix(c->o) * Vector4d(0, 0, -1, 1))(1);
  e1(2) = (makeRotationMatrix(c->o) * Vector4d(0, 0, -1, 1))(2);
  e1.normalize();

  e2(0) = (makeRotationMatrix(c->o) * Vector4d(1, 0, 0, 1))(0);
  e2(1) = (makeRotationMatrix(c->o) * Vector4d(1, 0, 0, 1))(1);
  e2(2) = (makeRotationMatrix(c->o) * Vector4d(1, 0, 0, 1))(2);
  e2.normalize();

  e3(0) = (makeRotationMatrix(c->o) * Vector4d(0, 1, 0, 1))(0);
  e3(1) = (makeRotationMatrix(c->o) * Vector4d(0, 1, 0, 1))(1);
  e3(2) = (makeRotationMatrix(c->o) * Vector4d(0, 1, 0, 1))(2);
  e3.normalize();

  for(int i = 0; i < x; i++){
    for(int j = 0; j < y; j++){
      ray r;
      r.origin(0) = camera_pos(0);
      r.origin(1) = camera_pos(1);
      r.origin(2) = camera_pos(2);

      r.direction = c->near * e1 + (i - x / 2) * 2 * c->right / x * e2 + e3 * (-j + y / 2) * 2 * c->top / y;
      r.direction.normalize();

      vertex pointHit;
      vertex normalHit;
      float minDistance = FLT_MAX;
      object *closestObj = NULL;
      for(int k = 0; k < objects.size(); k++){
        if(intersect(objects[k], r, &pointHit, &normalHit)){
          if(vertex_vector_distance(pointHit, r.origin) < minDistance){
            closestObj = objects[k];
            minDistance = vertex_vector_distance(pointHit, r.origin);
          }
        }
      }

      if(closestObj != NULL){
        vector<light*> lightsAtPixel;
        for(int k = 0; k < lights.size(); k++){
          ray shadowRay;
          bool shadow;
          shadowRay.origin(0) = pointHit.x;
          shadowRay.origin(1) = pointHit.y;
          shadowRay.origin(2) = pointHit.z;

          shadowRay.direction(0) = lights[k]->x - pointHit.x;
          shadowRay.direction(1) = lights[k]->y - pointHit.y;
          shadowRay.direction(2) = lights[k]->z - pointHit.z;

          shadowRay.direction.normalize();
          vertex point;
          vertex normal;
          for(int l = 0; l < objects.size(); l++){
              if(intersect(objects[l], shadowRay, &point, &normal)){
                if(vector_distance(vertexToVector(point), vertexToVector(pointHit)) < .1){
                  shadow = true;
                }
                break;
              }
          }
          if(!shadow){
            lightsAtPixel.push_back(lights[k]);
          }
        }
        pixels[j * x + i] = lighting(pointHit, normalHit, closestObj->m, lightsAtPixel, c->position);
      }
    }
  }
}
