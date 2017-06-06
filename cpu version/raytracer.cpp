#include "raytracer.h"

int main(int argc, char *argv[]){
	// Check that at least one file is provided to parse through
	if(argc != 4){
		cerr << "This program will only work if one file is provided and an xres and yres are provided" << endl;
	}
	else{

		int xres = atoi(argv[2]);
		int yres = atoi(argv[3]);

		// The ifstream object that parses the text file
		ifstream parser;
		parser.open(argv[1]);

		// Check that the file has been opened properly
		if(!parser.good()){
			cerr << "Something has gone wrong when opening your file. Please try again." << endl;
			return 1;
		}
		else{
			//Parse the file
			string buf;

      camera *c = new camera;
      vector<object *>objs;
      vector<object *>object_copies;
			//The first section of the file is the camera, so we will create a camera object and run our camera function
			parse_camera_data(&parser, c);

			vector<light*>lights = parse_light_data(&parser);
      cerr<<"num lights: "<<lights.size()<<endl;

			//This should be the objects: line
			getline(parser, buf);

			//This should be the first line with an .obj file
			getline(parser, buf);

			//The second section of the file is .obj files, so we will presume we are parsing through those lines
			//until we hit an empty line
			while((int)buf[0] != 0){
				stringstream ss;
				ss.clear();
				ss.str("");

				//Load the line into the stringstream
				ss<<buf;

				string name;
				string file_name;

				//Because the format of the line is name file_name, we can obtain them like this
				ss>>name>>file_name;
				ss.clear();
				ss.str("");
				//Give the file name the correct path
				ss<<"../data/"<<file_name;
				file_name = ss.str();
			    object *obj = new object;
			    obj->name = name;
			    if(parse_obj_file(file_name.c_str(), obj) == 1){
			    	return 1;
			    }
			    objs.push_back(obj);
			    getline(parser, buf);
			}

			while(!parser.eof()){
				// Read in a line
				string buf;
				getline(parser, buf);

				stringstream ss;

				ss.clear();
				ss.str("");

				//Load buf into the stringstream
				ss<<buf;

				//buf should be the object name, or a blank line.
				if((int)buf[0] != 0){
					int i;
					object *current_object;
					for(i = 0; i < objs.size(); i++){
						if(objs[i]->name.compare(buf) == 0){
							current_object = objs[i];
							break;
						}
					}
					//If the object exists, then record its transformations until another object is "hit" (empty line should be the indicator)
					if(current_object){
						//Create a new object that is a "copy" of the object from the .obj file and give that obj the transformations
						object *obj = new object;
						obj->name = current_object->name;
						obj->vertexes = current_object->vertexes;
						obj->normal_vertexes = current_object->normal_vertexes;
						obj->faces = current_object->faces;
						string s = find_material_reflectances(&parser, obj);
						find_transformations(&parser, obj, s);
            cerr<<obj->transformations.size()<<endl;
						object_copies.push_back(obj);
					}
				}
			}
			parser.close();

      //For each object to be transformed, we will tranform it and print it out to cout
			for(int i = 0; i < object_copies.size(); i++){
				//First add one to the "copy" count before printing
				vector<vertex*> transformed_vertexes;
				transformed_vertexes.push_back(NULL);

				//Determine the transformation matrix by left multiplying all matrices in the objects transformations vector
				vector <Matrix4d> transformations = object_copies[i]->transformations;
				Matrix4d m = Matrix4d::Identity();
				for(int n = 0; n < transformations.size(); n++){
          cerr<<"transformations"<<endl;
					m = transformations[n] * m;
				}

				//For each vertex (one-indexed), transform it
				for(int j = 1; j < object_copies[i]->vertexes.size(); j++){
					float x = object_copies[i]->vertexes[j]->x;
					float y = object_copies[i]->vertexes[j]->y;
					float z = object_copies[i]->vertexes[j]->z;
					float w = 1;

					Vector4d v(x, y, z, w);

					Vector4d transformed_v = m * v;

					//Create a new vertex reflecting the transformed vertex
					vertex *vtx = new vertex;
					vtx->x = transformed_v(0);
					vtx->y = transformed_v(1);
					vtx->z = transformed_v(2);

					transformed_vertexes.push_back(vtx);
				}
				//Replace the previous vertex vector with the transformed vertexes
				object_copies[i]->vertexes = transformed_vertexes;

				//Transform the normal vertexes in a similar fashion
				vector<vertex *>transformed_normals;
				transformed_normals.push_back(NULL);

				vector <Matrix4d> normal_transformations = object_copies[i]->normal_transformations;
				m = Matrix4d::Identity();
				for(int n = 0; n < normal_transformations.size(); n++){
					m = normal_transformations[n] * m;
				}

				Matrix4d normal_transformation = m.inverse();
				normal_transformation.transposeInPlace();
				for(int j = 1; j < object_copies[i]->normal_vertexes.size(); j++){
					float x = object_copies[i]->normal_vertexes[j]->x;
					float y = object_copies[i]->normal_vertexes[j]->y;
					float z = object_copies[i]->normal_vertexes[j]->z;
					float w = 1;

					Vector4d v(x, y, z, w);

					Vector4d transformed_n =  normal_transformation * v;

					//Create a new vertex reflecting the transformed normal and normalize it
					vertex *norm = new vertex;
					float mag = sqrt(transformed_n(0) * transformed_n(0) + transformed_n(1) * transformed_n(1) + transformed_n(2) * transformed_n(2));
					norm->x = transformed_n(0)/mag;
					norm->y = transformed_n(1)/mag;
					norm->z = transformed_n(2)/mag;

					transformed_normals.push_back(norm);
				}

				object_copies[i]->normal_vertexes = transformed_normals;
      }

      color pixels[xres * yres];
      for(int i = 0; i < xres * yres; i++){
        pixels[i].r = 0;
        pixels[i].g = 0;
        pixels[i].b = 0;
      }
      raytrace(object_copies, lights, c, pixels, xres, yres);

      cout<<"P3"<<endl;
			cout<<xres<<" "<<yres<<endl;
			cout<<"255"<<endl;

			for(int i = 0; i < yres; i++){
				for(int j = 0; j < xres; j++){
					cout<<(int)(pixels[i * xres + j].r)<<" "<<(int)(pixels[i * xres + j].g)<<" "<<(int)(pixels[i * xres + j].b)<<endl;
				}
			}
    }
  }
}
