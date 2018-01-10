#include "caffe/loss_layers.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <math.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	/*
	const Dtype* I = bottom[0]->cpu_data();
	const Dtype* f_x = bottom[1]->cpu_data();
	const Dtype* f_y = bottom[2]->cpu_data();
	//const Dtype* s = bottom[3]->cpu_data();
	*/
	//bottom[0] data
	//bottom[1] label


	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		CHECK(bottom[0]->num() == bottom[1]->num());
		this->_num = bottom[0]->num();
		this->_channel = bottom[0]->channels();
		this->_height = bottom[0]->height();
		this->_width = bottom[0]->width();
		
		this->local_feature_ = bottom[0];
		this->local_label_ = bottom[1];

        this->_margin_ident = this->layer_param().npair_loss_param().margin_ident();
        this->_margin_diff = this->layer_param().npair_loss_param().margin_diff();
        /*
        	identSN
        		Positive: 选择排序后的第identSN绝对顺位的距离做阈值
        		Negative: 选择排序后的-identSN*APNUM相对顺位的距离做阈值
        		应该还有一种配合策略：
        			是全局排序还是每个Query排序
        */
        this->_identSN = this->layer_param().npair_loss_param().identsn();
        this->_diffSN = this->layer_param().npair_loss_param().diffsn();

		if(Caffe::MULTI_GPU)
		{
			//Initialize MPI-Ref variables
			vector<int> local_feature_shape(this->local_feature_->shape());
			local_feature_shape[0] *= Caffe::NUM_GPU;
			this->total_feature_ = new Blob<Dtype>(local_feature_shape);

			vector<int> local_label_shape(this->local_label_->shape());
			local_label_shape[0] *= Caffe::NUM_GPU;
			this->total_label_ = new Blob<Dtype>(local_label_shape);

			vector<int> cross_compute_shape(2, 0);
			cross_compute_shape[0] = this->_num;
			cross_compute_shape[1] = Caffe::NUM_GPU * this->_num;
			this->_innerProd.Reshape(cross_compute_shape);
            this->_innerProd_calPrecision.Reshape(cross_compute_shape);
			// These variable for saving some temperaroy variable
			this->_innerProd_temp1.Reshape(cross_compute_shape);
			this->_innerProd_temp2.Reshape(cross_compute_shape);

			this->_query_diff_part1_weights.Reshape(cross_compute_shape);
			this->_query_diff_part2_weights.Reshape(cross_compute_shape);
			this->_query_diff_part3_weights.Reshape(cross_compute_shape);

			this->_isIdentType.Reshape(cross_compute_shape);
			this->_isDiffType.Reshape(cross_compute_shape);
			this->_cross_multiplier.Reshape(Caffe::NUM_GPU * this->_num, 1, 1, 1);
			this->_query_multiplier.Reshape(this->_num, 1, 1, 1);
			caffe_set(this->_cross_multiplier.count(), (Dtype)1, this->_cross_multiplier.mutable_cpu_data());
            caffe_set(this->_query_multiplier.count(), (Dtype)1, this->_query_multiplier.mutable_cpu_data());

			this->_isSelectPair.Reshape(cross_compute_shape);
			this->_tmp_Select_Ident.Reshape(cross_compute_shape);
			this->_tmp_Select_Diff.Reshape(cross_compute_shape);

			this->_min_within_dot_blob.Reshape(this->_num, 1, 1, 1);
			this->_max_between_dot_blob.Reshape(this->_num, 1, 1, 1);
			this->_max_all_dot_blob.Reshape(this->_num, 1, 1, 1);

			this->_nega_select_threshold.Reshape(this->_num, 1, 1, 1);
			this->_posi_select_threshold.Reshape(this->_num, 1, 1, 1);

			this->identNum.Reshape(this->_num, 1, 1, 1);
			this->diffNum.Reshape(this->_num, 1, 1, 1);
			this->loss_ident_value.Reshape(this->_num, 1, 1, 1);
			this->loss_diff_value.Reshape(this->_num, 1, 1, 1);

			this->_loss_value_tmp1_sum.Reshape(this->_num, 1, 1, 1);
			this->_loss_value_tmp2_div.Reshape(this->_num, 1, 1, 1);
			this->_loss_value_tmp3_log.Reshape(this->_num, 1, 1, 1);


			this->loss_scalar_part1.Reshape(this->_num, 1, 1, 1);
			this->loss_scalar_part2.Reshape(this->_num, 1, 1, 1);
			this->loss_scalar_part3.Reshape(this->_num, 1, 1, 1);
		}
		else{
			// Standard mode (ORIGINAL)//not supported
			//Initialize MPI-Ref variables
			vector<int> local_feature_shape(this->local_feature_->shape());
			local_feature_shape[0] *= Caffe::NUM_GPU;
			this->total_feature_ = new Blob<Dtype>(local_feature_shape);

			vector<int> local_label_shape(this->local_label_->shape());
			local_label_shape[0] *= Caffe::NUM_GPU;
			this->total_label_ = new Blob<Dtype>(local_label_shape);

			vector<int> cross_compute_shape(2, 0);
			cross_compute_shape[0] = this->_num;
			cross_compute_shape[1] = Caffe::NUM_GPU * this->_num;
			this->_innerProd.Reshape(cross_compute_shape);
            this->_innerProd_calPrecision.Reshape(cross_compute_shape);
			// These variable for saving some temperaroy variable
			this->_innerProd_temp1.Reshape(cross_compute_shape);
			this->_innerProd_temp2.Reshape(cross_compute_shape);

			this->_query_diff_part1_weights.Reshape(cross_compute_shape);
			this->_query_diff_part2_weights.Reshape(cross_compute_shape);
			this->_query_diff_part3_weights.Reshape(cross_compute_shape);

			this->_isIdentType.Reshape(cross_compute_shape);
			this->_isDiffType.Reshape(cross_compute_shape);
			this->_cross_multiplier.Reshape(Caffe::NUM_GPU * this->_num, 1, 1, 1);
			this->_query_multiplier.Reshape(this->_num, 1, 1, 1);
			caffe_set(this->_cross_multiplier.count(), (Dtype)1, this->_cross_multiplier.mutable_cpu_data());
            caffe_set(this->_query_multiplier.count(), (Dtype)1, this->_query_multiplier.mutable_cpu_data());
			
			this->_isSelectPair.Reshape(cross_compute_shape);
			this->_tmp_Select_Ident.Reshape(cross_compute_shape);
			this->_tmp_Select_Diff.Reshape(cross_compute_shape);
			
			this->_min_within_dot_blob.Reshape(this->_num, 1, 1, 1);
			this->_max_between_dot_blob.Reshape(this->_num, 1, 1, 1);
			this->_max_all_dot_blob.Reshape(this->_num, 1, 1, 1);

			this->_nega_select_threshold.Reshape(this->_num, 1, 1, 1);
			this->_posi_select_threshold.Reshape(this->_num, 1, 1, 1);

			this->identNum.Reshape(this->_num, 1, 1, 1);
			this->diffNum.Reshape(this->_num, 1, 1, 1);
			this->loss_ident_value.Reshape(this->_num, 1, 1, 1);
			this->loss_diff_value.Reshape(this->_num, 1, 1, 1);

			this->_loss_value_tmp1_sum.Reshape(this->_num, 1, 1, 1);
			this->_loss_value_tmp2_div.Reshape(this->_num, 1, 1, 1);
			this->_loss_value_tmp3_log.Reshape(this->_num, 1, 1, 1);

			this->loss_scalar_part1.Reshape(this->_num, 1, 1, 1);
			this->loss_scalar_part2.Reshape(this->_num, 1, 1, 1);
			this->loss_scalar_part3.Reshape(this->_num, 1, 1, 1);
		}
	}

	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> shape(0);
        for(int i=0; i<top.size(); i++){
            top[i]->Reshape(shape);
        }
        /*
		top[0]->Reshape(shape);//loss
		top[1]->Reshape(shape);//top-1 retrieve accuracy
		top[2]->Reshape(shape);//top-5 retrieve accuracy
        */
	}


	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	}



	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}


#ifdef CPU_ONLY
STUB_GPU(NPairMultiClassLossLayer);
#endif
	INSTANTIATE_CLASS(NPairMultiClassLossLayer);
	REGISTER_LAYER_CLASS(NPairMultiClassLoss);
}
