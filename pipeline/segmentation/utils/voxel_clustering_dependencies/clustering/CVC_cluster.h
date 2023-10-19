#ifndef CVC_CLUSTER_H
#define CVC_CLUSTER_H

#include <iostream>
#include <stdlib.h>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <vector>
#include <cmath> 
#include <Eigen/Dense>

namespace cvc{



struct PointXYZ {
    float x;
    float y;
    float z;

    PointXYZ(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};


struct PointAPR{
   float azimuth;
   float polar_angle;
   float range;
};

struct Voxel{
   bool haspoint = false;
   int cluster = -1;
   std::vector<int> index;
};



class CVC{
	public:
	CVC(){}
	
	CVC(std::vector<float>& param){
		if(param.size() != 3){
			printf("Param number is not correct!");
			std::abort();		
		}
		for(int i=0; i<param.size(); ++i){
			deltaA_ = param[0];
			deltaR_ = param[1];
			deltaP_ = param[2];
		}
	}

	~CVC(){}
	std::vector<PointAPR> calculateAPR(Eigen::MatrixXf cloud_IN);
	std::unordered_map<int, Voxel> build_hash_table(const std::vector<PointAPR>& vapr);
	void find_neighbors(int polar, int range, int azimuth, std::vector<int>& neighborindex);
	std::vector<int> most_frequent_value(std::vector<int> values);
	void mergeClusters(std::vector<int>& cluster_indices, int idx1, int idx2);
	std::vector<int>  cluster(std::unordered_map<int, Voxel> &map_in,const std::vector<PointAPR>& vapr);
	//void process();

private:
	float deltaA_ = 2;
	float deltaR_ = 0.35;
	float deltaP_ = 1.2;
	float min_range_ = std::numeric_limits<float>::max();
	float max_range_ = std::numeric_limits<float>::min();

	
	float min_azimuth_ = -24.8 * M_PI/180;
	float max_azimuth_ = 2 * M_PI/180;
	int length_ = 0;
	int width_  = 0;
	int height_ = 0;
};

};

#endif


