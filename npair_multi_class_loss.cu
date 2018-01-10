#include "caffe/loss_layers.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <cfloat>

#include <cmath>
#include <math.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::GatherFeatureAndLabel()
	{
		int this_query_num = this->_num;
		int this_database_num = this->_num * Caffe::NUM_GPU; 
		int fea_dim = this->_channel * this->_height * this->_width;
		  int count = this_query_num * fea_dim;//this->local_feature_->count();
		  const Dtype * sendbuf = this->local_feature_->cpu_data();
		  Dtype * recvbuf = this->total_feature_->mutable_cpu_data();
		  //LOG(INFO)<<"LOG_0";
		  int count_label = this->local_label_->count();
		  const Dtype * send_label = this->local_label_->cpu_data();
		  Dtype * recv_label = this->total_label_->mutable_cpu_data();
		  
		  if (sizeof(Dtype) == 4) {
		    //MPI_Gather(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
		    //MPI_Bcast(recvbuf, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
		    MPI_Allgather(sendbuf, count, MPI_FLOAT, recvbuf, count, MPI_FLOAT, MPI_COMM_WORLD);
		    //LOG(INFO) << "All-gather feature ok...";
		    MPI_Allgather(send_label, count_label, MPI_FLOAT, recv_label, count_label, MPI_FLOAT, MPI_COMM_WORLD);
		    //LOG(INFO)<<"LOG_2_1";
		    //LOG(INFO) << "All-gather label ok...";
		  } else if(sizeof(Dtype) == 8) {
		    MPI_Allgather(sendbuf, count, MPI_DOUBLE, recvbuf, count, MPI_DOUBLE, MPI_COMM_WORLD);
		    MPI_Allgather(send_label, count_label, MPI_DOUBLE, recv_label, count_label, MPI_DOUBLE, MPI_COMM_WORLD);
		    //LOG(INFO)<<"LOG_2_2";
		  } else {
		    LOG(FATAL) << "Error size of Dtype: " << sizeof(Dtype);
		  }   
	}
	template <typename Dtype>
	__global__ void GetLabelDiffMtx(const int nthreads, const int local_label_num, const int database_label_num,const int local_gpu_id, const Dtype* local_labels, const Dtype* database_labels, Dtype* same_mtx, Dtype* diff_mtx){
		CUDA_KERNEL_LOOP(index, nthreads) {
			same_mtx[index] = (Dtype)0;
			diff_mtx[index] = (Dtype)0;

			const int database_lbl_idx = index % database_label_num;
			const int local_lbl_idx = index / database_label_num;
			Dtype local_lbl = local_labels[local_lbl_idx];
			Dtype database_lbl = database_labels[database_lbl_idx];
			if(local_lbl_idx + local_gpu_id * local_label_num != database_lbl_idx){
				if(local_lbl == database_lbl){
					same_mtx[index] = (Dtype)1;
					diff_mtx[index] = (Dtype)0;
				}
				else{
					same_mtx[index] = (Dtype)0;
					diff_mtx[index] = (Dtype)1;
				}
			}
			
		}
	}


	template <typename Dtype>
	__global__ void GetSampledPairMtx(const int nthreads, const int local_label_num, const int database_label_num,const int local_gpu_id, 
		const Dtype* local_labels, const Dtype* database_labels, const Dtype* same_mtx, const Dtype* diff_mtx, 
		const Dtype* posi_thresholds, const int posi_select_method, const Dtype* nega_thresholds, const int nega_select_method,
		const Dtype* innerProd, Dtype* is_select_pair_mtx, const Dtype margin_ident, const Dtype margin_diff){
		CUDA_KERNEL_LOOP(index, nthreads) {
			int query_idx = index / database_label_num;
			int base_idx = index % database_label_num;

			is_select_pair_mtx[index] = 0;
			if(same_mtx[index] == 1) {
				/*
				//正样本全选，不挑
                //if(same_mtx[index] > (max_between_dot[query_idx])
                if(innerProd[index] > (max_between_dot[query_idx] + min_within_dot[query_idx]) / 2)
				    is_select_pair_mtx[index] = 1;
				    */
				if(posi_select_method == 0){ //HARD
					if(innerProd[index] < posi_thresholds[query_idx] + margin_ident){
						is_select_pair_mtx[index] = 1;
					}
				}else if(posi_select_method == 1) {//EASY
					if(innerProd[index] >= posi_thresholds[query_idx] + margin_ident){
						is_select_pair_mtx[index] = 1;
					}
				}else if(posi_select_method == 2) {//ALL
					is_select_pair_mtx[index] = 1;
				}else if(posi_select_method == 3) {//RELATIVE_HARD
					if(innerProd[index] <= posi_thresholds[query_idx] + margin_ident){
						is_select_pair_mtx[index] = 1;
					}
				}else if(posi_select_method == 4) {//RELATIVE_EASY
					if(innerProd[index] >= posi_thresholds[query_idx] + margin_ident){
						is_select_pair_mtx[index] = 1;
					}
				}
			}
			else if(diff_mtx[index] == 1) {
				/*
                //is_select_pair_mtx[index] = 1;
                
                
				//负样本选择内积大于 最小正样本内积

				if(innerProd[index] > min_within_dot[query_idx]) { 
				//if(diff_mtx[index] > min_within_dot[query_idx] - semi_margin && diff_mtx[index] < min_within_dot[query_idx] + semi_margin){
				//if(diff_mtx[index] < min_within_dot[query_idx]){ // - semi_margin && diff_mtx[index] < min_within_dot[query_idx] + semi_margin){
					is_select_pair_mtx[index] = 1;
				}
				*/
				if(nega_select_method == 0){ //HARD
					if(innerProd[index] > nega_thresholds[query_idx] + margin_diff){
						is_select_pair_mtx[index] = 1;
					}
				}else if(nega_select_method == 1) {//EASY
					if(innerProd[index] <= nega_thresholds[query_idx] + margin_diff){
						is_select_pair_mtx[index] = 1;
					}
				}else if(nega_select_method == 2) {//ALL
					is_select_pair_mtx[index] = 1;
				}else if(nega_select_method == 3) {//RELATIVE_HARD
					if(innerProd[index] >= nega_thresholds[query_idx] + margin_diff){
						is_select_pair_mtx[index] = 1;
					}
				}else if(nega_select_method == 4) {//RELATIVE_EASY
					if(innerProd[index] <= nega_thresholds[query_idx] + margin_diff){
						is_select_pair_mtx[index] = 1;
					}
				}
			}
		}
	}

	template <typename Dtype>
	__global__ void Minus_Querywise_Maxval(const int nthreads, const int local_label_num, const int database_label_num,
				const Dtype* _max_all_dot, const Dtype* identNum, const Dtype* diffNum, const Dtype* isIdent, const Dtype* isDiff, Dtype* innerProd, Dtype* innerProd_calPrecision) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int query_idx = index / database_label_num;
			int base_idx = index % database_label_num;
			innerProd[index] -= _max_all_dot[query_idx];
			innerProd[index] = expf(innerProd[index]);
            innerProd_calPrecision[index] = innerProd[index];
			if(isIdent[index] == 1){
				if(identNum[query_idx] == 0){
					innerProd[index] = 0;//因为相同这块是除数，所以赋值为0就可以，不产生LOSS
				}
				else{
                    //草他妈，这里不应该归一化，真日狗了
					innerProd[index] /= 1;//identNum[query_idx];
				}
			}
			else if(isDiff[index] == 1){
				if(diffNum[query_idx] == 0){
                    //为了去除
					innerProd[index] = 0;
				}
				else{
                    //草他妈，这里不应该归一化，真日狗了
					innerProd[index] /= 1;//diffNum[query_idx];
				}
			}
			else{
				innerProd[index] = 0;
			}
		}
	}

	template <typename Dtype>
	__global__ void ManipulateDIVandLOG(const int nthreads, const Dtype* identSumValue, const Dtype* allSumValue,
			  Dtype* divValue, Dtype* logValue) {
		CUDA_KERNEL_LOOP(index, nthreads) {
            if(identSumValue[index]==0 || allSumValue[index] == 0) {
                divValue[index] = 0;//identSumValue[index] / allSumValue[index];
                logValue[index] = 0;
            }
            else{
                divValue[index] = identSumValue[index] / allSumValue[index];
                logValue[index]  = logf(divValue[index]);
            }
	    }
    }
	/*
	template <typename Dtype>
	bool NPairMultiClassLossLayer<Dtype>::comp(Dtype a, Dtype b){
		return a > b;
	}
	*/

	template <typename Dtype>
	Dtype NPairMultiClassLossLayer<Dtype>::GetRetrivePerformance(const int query_num, const int global_num, const int gpu_rank, const Dtype* distanceMtx, const Dtype* query_labels, const Dtype* global_labels, const int top_k){

		int retrived_query_num = 0;
		for(int q_idx = 0; q_idx < query_num; q_idx++){
			vector<Dtype> _database_pair_dist;
			const int this_query_dist_begin_idx = q_idx * global_num;
			for(int dp_dist_idx = 0; dp_dist_idx < global_num; dp_dist_idx++) {
				// 如果不是跟自己比较，就都放到list里
				if(gpu_rank * query_num + q_idx != dp_dist_idx) {
					_database_pair_dist.push_back(distanceMtx[this_query_dist_begin_idx + dp_dist_idx]);
				}
			}
			//因为这里使用内积作为度量，所以这个距离是越大越相似，因为我们是找到top-k个最像的，所以从大往小排序
			//最近的肯定是自己，所以top-k我们选择第k+1个作为阈值
			std::sort(_database_pair_dist.begin(), _database_pair_dist.end(), this->comp);
            /*
            LOG(INFO)<<"--------------------------------top:"<<top_k;
            LOG(INFO)<<"SELF DIST:" << distanceMtx[this_query_dist_begin_idx + gpu_rank * query_num + q_idx];
            
            for(int i=0;i<4 ;i++){ //_database_pair_dist.size();i++){
                LOG(INFO)<<i<<":"<<_database_pair_dist[i];
            }
            */
			Dtype threshold = _database_pair_dist[std::min(top_k, (int)_database_pair_dist.size() - 1)];
			Dtype max = _database_pair_dist[0], min = _database_pair_dist[_database_pair_dist.size()-1];


			for(int dp_dist_idx = 0; dp_dist_idx < global_num; dp_dist_idx++) {
				// 如果不是跟自己比较，计算精度
				if(gpu_rank * query_num + q_idx != dp_dist_idx) {
					if(distanceMtx[this_query_dist_begin_idx + dp_dist_idx] > threshold && query_labels[q_idx] == global_labels[dp_dist_idx]) {
						retrived_query_num ++;
						//LOG(INFO) << " ,Max_value:" << max << " ,Min_value:"<< min << " ,Threshold:" << threshold << " ,QUERY IN GLOBAL IDX:" << gpu_rank * query_num + q_idx << " , HITTED DATABASE IDX:" << dp_dist_idx << " , HITTED DISTANCE:" << distanceMtx[this_query_dist_begin_idx + dp_dist_idx] << " ,HITTED LABEL:" << global_labels[dp_dist_idx];
						break;
					}
				}
			}
		}
		return (Dtype)retrived_query_num / (Dtype)query_num;
	}
	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		/*
		this->_num = bottom[0]->num();
		this->_channel = bottom[0]->channels();
		this->_height = bottom[0]->height();
		this->_width = bottom[0]->width();
		this->local_feature_ = bottom[0];
		this->local_label_ = bottom[1];
		this->total_feature_ = new Blob<Dtype>(local_feature_shape);
		this->total_label_ = new Blob<Dtype>(local_label_shape);
		*/
		//LOG(INFO)<<"FORWARD STEP1 BEGIN";
		GatherFeatureAndLabel();
		//LOG(INFO)<<"FORWARD STEP2 BEGIN";
		int this_query_num = this->_num;
		int this_database_num = this->_num * Caffe::NUM_GPU; 
		int fea_dim = this->_channel * this->_height * this->_width;
        int dot_normalizer = 1;//fea_dim;

		caffe_gpu_gemm(CblasNoTrans, CblasTrans, this_query_num, this_database_num, fea_dim, (Dtype)1 / dot_normalizer, this->local_feature_->gpu_data(), this->total_feature_->gpu_data(), (Dtype)0, this->_innerProd.mutable_gpu_data());
        /*
        for(int i=0;i<this->_innerProd.count();i++)
        {
            if(this->_innerProd.cpu_data()[i] != 0){
                LOG(INFO) <<i<< "INNER_PROD:" << this->_innerProd.cpu_data()[i];
            }
        }
        */
		

		/*
		for(int i=0;i<100;i++){
			LOG(INFO) <<i<< ",LOCAL_FEATURE:" << this->local_feature_->cpu_data()[i]
				<< "GLOBAL_FEATURE:" << this->total_feature_->cpu_data()[i]
				<< "CROSS_DIST_MTX:" << this->_innerProd.cpu_data()[i];
		}
		*/



		GetLabelDiffMtx<Dtype><<<CAFFE_GET_BLOCKS(this_query_num * this_database_num), CAFFE_CUDA_NUM_THREADS>>>(this_query_num * this_database_num, this_query_num, this_database_num, Caffe::RANK, this->local_label_->gpu_data(), this->total_label_->gpu_data(), this->_isIdentType.mutable_gpu_data(), this->_isDiffType.mutable_gpu_data());
		
		
		// Not normalize the distance
		//identNum = 1;
		//diffNum = 1;
		//LOG(INFO)<<"FORWARD STEP3 BEGIN";
		//Dtype _max_within_dot = -FLT_MAX;
		//Dtype _max_between_dot = -FLT_MAX;

		//数值统计
		/* old数值统计，有问题，应该分类别进行统计
		Dtype _min_within_dot = FLT_MAX;
		Dtype _max_between_dot = -FLT_MAX;
		Dtype _max_all_dot = -FLT_MAX;
		for(int i = 0; i < this_query_num * this_database_num; i++){
			if( this->_isIdentType.cpu_data()[i] == 1 ) {
				if(this->_innerProd.cpu_data()[i] < _min_within_dot){
					_min_within_dot = this->_innerProd.cpu_data()[i];
				}
				if(_innerProd.cpu_data()[i] > _max_all_dot) {
					_max_all_dot = _innerProd.cpu_data()[i];
				}
			}
			else if( this->_isDiffType.cpu_data()[i] == 1 ) {
				if(this->_innerProd.cpu_data()[i] > _max_between_dot) {
					_max_between_dot = this->_innerProd.cpu_data()[i];
				}
				if(_innerProd.cpu_data()[i] > _max_all_dot) {
					_max_all_dot = _innerProd.cpu_data()[i];
				}
			}
			
		}
		*/

		//新的数值统计20171225
		//Dtype* _min_with_dot;
		//全部的相同pair距离，全部的不同pair距离
		vector<Dtype> ident_prod_global_list, diff_prod_global_list;
		//分query的相同pair距离，分query的不同pair距离
		vector< vector<Dtype> > ident_prod_local_list, diff_prod_local_list;
		//注意，这个变量可以继续优化
		Dtype* _max_all_dot = this->_max_all_dot_blob.mutable_cpu_data();
		caffe_set(this_query_num, (Dtype)-FLT_MAX, _max_all_dot);
		//长度为 this_query_num,表示query这个类内部的类内最小内积(内积越大越相似)
		Dtype* _min_within_dot = _min_within_dot_blob.mutable_cpu_data();
		caffe_set(this_query_num, (Dtype)FLT_MAX, _min_within_dot);
		//长度为 this_query_num,表示query这个类内部的类间最大内积(内积越小越相似)
		Dtype* _max_between_dot = _max_between_dot_blob.mutable_cpu_data();
		caffe_set(this_query_num, (Dtype)-FLT_MAX, _max_between_dot);
		for(int query_idx = 0 ;query_idx < this_query_num; query_idx++) {
			vector<Dtype> ident_prod_local_queryi_list, diff_prod_local_queryi_list;
			for( int base_idx = 0; base_idx < this_database_num; base_idx++) {
				int this_idx = query_idx * this_database_num + base_idx;

				if( this->_isIdentType.cpu_data()[this_idx] == 1 ) {
					if(this->_innerProd.cpu_data()[this_idx] < _min_within_dot[query_idx]){
						_min_within_dot[query_idx] = this->_innerProd.cpu_data()[this_idx];
					}
					if(_innerProd.cpu_data()[this_idx] > _max_all_dot[query_idx]) {
						_max_all_dot[query_idx] = _innerProd.cpu_data()[this_idx];
					}
					ident_prod_local_queryi_list.push_back(_innerProd.cpu_data()[this_idx]);
					ident_prod_global_list.push_back(_innerProd.cpu_data()[this_idx]);
				}
				else if( this->_isDiffType.cpu_data()[this_idx] == 1 ) {
					if(this->_innerProd.cpu_data()[this_idx] > _max_between_dot[query_idx]) {
						_max_between_dot[query_idx] = this->_innerProd.cpu_data()[this_idx];
					}
					if(_innerProd.cpu_data()[this_idx] > _max_all_dot[query_idx]) {
						_max_all_dot[query_idx] = _innerProd.cpu_data()[this_idx];
					}
					diff_prod_local_queryi_list.push_back(_innerProd.cpu_data()[this_idx]);
					diff_prod_global_list.push_back(_innerProd.cpu_data()[this_idx]);
				}
			}
			ident_prod_local_list.push_back(ident_prod_local_queryi_list);
			diff_prod_local_list.push_back(diff_prod_local_queryi_list);
		}
		//GLOBAL距离排序（升序）
		std::sort(ident_prod_global_list.begin(), ident_prod_global_list.end());
		std::sort(diff_prod_global_list.begin(), diff_prod_global_list.end());
		//LOCAL距离排序（升序）
		for(int query_idx = 0; query_idx < this_query_num; query_idx++) {
			std::sort(ident_prod_local_list[query_idx].begin(), ident_prod_local_list[query_idx].end());
			std::sort(diff_prod_local_list[query_idx].begin(), diff_prod_local_list[query_idx].end());
		}

		//根据配置选择正样本、负样本的分界值点（可能仍不够灵活）
		//AP选择
		if(this->layer_param_.npair_loss_param().ap_mining_region() == NPairLossParameter_MiningRegion_LOCAL) {
			if(this->layer_param_.npair_loss_param().ap_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_EASY && this->layer_param_.npair_loss_param().ap_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_HARD) { // 根据绝对位置选择（难/易/随机）
				caffe_cpu_axpby(this->_max_between_dot_blob.count(), (Dtype)1, this->_max_between_dot_blob.cpu_data(), (Dtype)0, this->_posi_select_threshold.mutable_cpu_data());
			}
			else {
				for(int i = 0; i < ident_prod_local_list.size(); i++){
					int pos = 0;
					//identSN为从大往小排（从易往难），先选容易的
					pos = (this->_identSN >= 0) ?
						 (int)( ident_prod_local_list[i].size() - 1 - (int)this->_identSN) :
						 (int)(ident_prod_local_list[i].size() - 1 + this->_identSN * ident_prod_local_list[i].size());
					Dtype thre = ident_prod_local_list[i][pos]>=0? ident_prod_local_list[i][pos] : -FLT_MAX;
					this->_posi_select_threshold.mutable_cpu_data()[i] = thre;
				}
			}
		}
		else if(this->layer_param_.npair_loss_param().ap_mining_region() == NPairLossParameter_MiningRegion_GLOBAL) {
			if(this->layer_param_.npair_loss_param().ap_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_EASY  && this->layer_param_.npair_loss_param().ap_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_HARD) {// 根据绝对位置选择（难/易/随机）
				//把全局最大类间距离赋给正样本选择阈值
				caffe_set(this->_posi_select_threshold.count(), (Dtype)diff_prod_global_list[diff_prod_global_list.size() - 1],  this->_posi_select_threshold.mutable_cpu_data());
			}
			else {
				//identSN为从大往小排（从易往难），先选容易的
				int pos = (this->_identSN >= 0) ?
					int( ident_prod_global_list.size() - 1 - (int)this->_identSN):
					(int)(ident_prod_global_list.size() - 1 + this->_identSN * ident_prod_global_list.size());
				Dtype thre = ident_prod_global_list[pos] >=0 ? ident_prod_global_list[pos] : -FLT_MAX;
				caffe_set( this->_posi_select_threshold.count(), thre,  this->_posi_select_threshold.mutable_cpu_data() );
			}
		}
		//AN选择
		if(this->layer_param_.npair_loss_param().an_mining_region() == NPairLossParameter_MiningRegion_LOCAL) {
			if(this->layer_param_.npair_loss_param().an_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_EASY && this->layer_param_.npair_loss_param().an_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_HARD) { // 根据绝对位置选择（难/易/随机）
				caffe_cpu_axpby(this->_min_within_dot_blob.count(), (Dtype)1, this->_min_within_dot_blob.cpu_data(), (Dtype)0, this->_nega_select_threshold.mutable_cpu_data());
			}
			else {
				for(int i = 0; i < diff_prod_local_list.size(); i++){
					int pos = 0;
					//diffSN为从大往小排（从难往易），先选难的
					pos = (this->_diffSN >= 0) ?
						 int( diff_prod_local_list[i].size() - 1 - (int)this->_diffSN):
						(int)(diff_prod_local_list[i].size() - 1 + this->_diffSN * diff_prod_local_list[i].size());
					Dtype thre = diff_prod_local_list[i][pos]>=0? diff_prod_local_list[i][pos] : -FLT_MAX;
					this->_nega_select_threshold.mutable_cpu_data()[i] = thre;
				}
			}
		}
		else if(this->layer_param_.npair_loss_param().an_mining_region() == NPairLossParameter_MiningRegion_GLOBAL) {
			if(this->layer_param_.npair_loss_param().an_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_EASY && this->layer_param_.npair_loss_param().an_mining_method() != NPairLossParameter_MiningMethod_RELATIVE_HARD) {// 根据绝对位置选择（难/易/随机）
				//把全局最小类内距离赋给负样本选择阈值
				caffe_set(this->_nega_select_threshold.count(), (Dtype)ident_prod_global_list[0],  this->_nega_select_threshold.mutable_cpu_data());
			}
			else {
				//diffSN为从大往小排（从难往易），先选难的
				int pos = (this->_diffSN >= 0) ?
					(int)( diff_prod_global_list.size() - 1 - (int)this->_diffSN):
					(int)(diff_prod_global_list.size() - 1 + this->_diffSN * diff_prod_global_list.size());
				Dtype thre = diff_prod_global_list[pos] >= 0? diff_prod_global_list[pos] : -FLT_MAX;
				caffe_set( this->_nega_select_threshold.count(), thre,  this->_nega_select_threshold.mutable_cpu_data() );
			}
		}

        //这是Normalizer
		//caffe_set(this_query_num, (Dtype)0, _max_all_dot);

		//根据一些统计量进行样本选择，记住，内积是越大表示两个向量越相近
		GetSampledPairMtx<Dtype><<<CAFFE_GET_BLOCKS(this_query_num * this_database_num), CAFFE_CUDA_NUM_THREADS>>>
			(this_query_num * this_database_num, this_query_num, this_database_num, Caffe::RANK,
			 this->local_label_->gpu_data(), this->total_label_->gpu_data(), this->_isIdentType.gpu_data(),
			  this->_isDiffType.gpu_data(),
			  _posi_select_threshold.gpu_data(), (int)this->layer_param_.npair_loss_param().ap_mining_method(),
			  _nega_select_threshold.gpu_data(), (int)this->layer_param_.npair_loss_param().an_mining_method(),
			  this->_innerProd.gpu_data(), this->_isSelectPair.mutable_gpu_data(), this->_margin_ident, this->_margin_diff);//这里有一个新变量


		//这两个变量改为类内局部变量
		
		//identNum = 0, diffNum = 0;
		caffe_gpu_mul(this_query_num * this_database_num, this->_isIdentType.gpu_data(), this->_isSelectPair.gpu_data(), this->_tmp_Select_Ident.mutable_gpu_data());
		//caffe_gpu_dot(this_query_num * this_database_num, this->_tmp_Select_Ident.gpu_data(), this->_cross_multiplier.gpu_data(), &identNum);
		caffe_gpu_gemv(CblasNoTrans, this_query_num, this_database_num, (Dtype)1, this->_tmp_Select_Ident.gpu_data(), this->_cross_multiplier.gpu_data(), (Dtype)0, this->identNum.mutable_gpu_data());
		caffe_gpu_mul(this_query_num * this_database_num, this->_isDiffType.gpu_data(), this->_isSelectPair.gpu_data(), this->_tmp_Select_Diff.mutable_gpu_data());
		//caffe_gpu_dot(this_query_num * this_database_num, this->_tmp_Select_Diff.gpu_data(), this->_cross_multiplier.gpu_data(), &diffNum);
		caffe_gpu_gemv(CblasNoTrans, this_query_num, this_database_num, (Dtype)1, this->_tmp_Select_Diff.gpu_data(), this->_cross_multiplier.gpu_data(), (Dtype)0, this->diffNum.mutable_gpu_data());
		

    /*
        LOG(INFO)<<"----------------";



        for(int i=0;i<this_query_num;i++){
            
            LOG(INFO)<< "query["<<i<<"], IDENTNUM:"<<this->identNum.cpu_data()[i]<<",DIFFNUM:"<<this->diffNum.cpu_data()[i]<< ",POSITHRE:"<<this->_posi_select_threshold.cpu_data()[i] << ",NEGATHRE"<<this->_nega_select_threshold.cpu_data()[i];
        }
    */
		Dtype loss = 0;
		if(true) {//(identNum != 0 && diffNum != 0) {



			// Minus Max for overcoming digit overflow
			Minus_Querywise_Maxval<Dtype><<<CAFFE_GET_BLOCKS(this_query_num * this_database_num), CAFFE_CUDA_NUM_THREADS>>>(this_query_num * this_database_num, this_query_num, this_database_num, 
				_max_all_dot_blob.gpu_data(), this->identNum.gpu_data(), this->diffNum.gpu_data(), this->_isIdentType.gpu_data(), this->_isDiffType.gpu_data(), this->_innerProd.mutable_gpu_data(), this->_innerProd_calPrecision.mutable_gpu_data());

            
            /*
            for(int i=0;i<4;i++)
            {
                LOG(INFO)<<"INNERPROD"<<i<<"="<<_innerProd_calPrecision.cpu_data()[i];
            }
            */
            
            
			//caffe_gpu_add_scalar(this_query_num * this_database_num, (Dtype)(-_max_all_dot), this->_innerProd.mutable_gpu_data());
			/*

			for(int i=0;i< 100;i++){
				LOG(INFO)<<"innerprod- step2["<<i<<"]: "<<_max_all_dot << "    "<<this->_innerProd.cpu_data()[i];
			}
			*/
			//caffe_gpu_exp(this_query_num * this_database_num, this->_innerProd.gpu_data(), this->_innerProd.mutable_gpu_data());
			/*
			for(int i=0;i< 100;i++){
				LOG(INFO)<<"innerprod- step3["<<i<<"]: "<<this->_innerProd.cpu_data()[i];
			}
			*/
			//这两个变量改为类内局部变量
			//loss_ident_value = 0, loss_diff_value = 0;
			caffe_gpu_mul(this_query_num * this_database_num, this->_innerProd.gpu_data(), this->_tmp_Select_Ident.gpu_data(), _innerProd_temp1.mutable_gpu_data());
			//caffe_gpu_dot(this_query_num * this_database_num, this->_innerProd_temp1.gpu_data(), this->_cross_multiplier.gpu_data(), &loss_ident_value);
			caffe_gpu_gemv(CblasNoTrans, this_query_num, this_database_num, (Dtype)1, this->_innerProd_temp1.gpu_data(), this->_cross_multiplier.gpu_data(), (Dtype)0, this->loss_ident_value.mutable_gpu_data());
			caffe_gpu_mul(this_query_num * this_database_num, this->_innerProd.gpu_data(), this->_tmp_Select_Diff.gpu_data(), _innerProd_temp2.mutable_gpu_data());
			//caffe_gpu_dot(this_query_num * this_database_num, this->_innerProd_temp2.gpu_data(), this->_cross_multiplier.gpu_data(), &loss_diff_value);
			caffe_gpu_gemv(CblasNoTrans, this_query_num, this_database_num, (Dtype)1, this->_innerProd_temp2.gpu_data(), this->_cross_multiplier.gpu_data(), (Dtype)0, this->loss_diff_value.mutable_gpu_data());

			caffe_gpu_add(this_query_num, this->loss_ident_value.gpu_data(), this->loss_diff_value.gpu_data(), this->_loss_value_tmp1_sum.mutable_gpu_data());
            /*
            for(int i=0;i<4;i++)
            {
                LOG(INFO)<<i<<"identval:"<<this->loss_ident_value.cpu_data()[i]<<",diffval:"<<this->loss_diff_value.cpu_data()[i];
            }
            */
            ManipulateDIVandLOG<Dtype><<<CAFFE_GET_BLOCKS(this_query_num), CAFFE_CUDA_NUM_THREADS>>>(this_query_num, this->loss_ident_value.gpu_data(), this->_loss_value_tmp1_sum.gpu_data(), this->_loss_value_tmp2_div.mutable_gpu_data(),
                this->_loss_value_tmp3_log.mutable_gpu_data());
            /*
			caffe_gpu_div(this_query_num, this->loss_ident_value.gpu_data(), this->_loss_value_tmp1_sum.gpu_data(), this->_loss_value_tmp2_div.mutable_gpu_data());

			caffe_gpu_log(this_query_num, this->_loss_value_tmp2_div.gpu_data(), this->_loss_value_tmp3_log.mutable_gpu_data());
            */

			//LOG(INFO)<<"FORWARD STEP4 BEGIN";
			//LOG(INFO)<<"loss_ident_value:"<<loss_ident_value<<",identNum:"<<identNum<<",loss_diff_value:"<<loss_diff_value<<",diffNum:"<<diffNum;
			//loss = -log( (loss_ident_value / identNum) / (loss_ident_value / identNum + loss_diff_value / diffNum) );
			caffe_gpu_dot(this_query_num, this->_loss_value_tmp3_log.gpu_data(), this->_query_multiplier.gpu_data(), &loss );
			loss /= -this_query_num;
		}
		/*
		GetRetrivePerformance(const int query_num, const int global_num, const int query_begin_idx_in_global, const Dtype* distanceMtx, const Dtype* query_labels, const Dtype* global_labels, const int top_k)
		*/
        /*
		Dtype top_1_recall = GetRetrivePerformance(this_query_num, this_database_num, Caffe::RANK , this->_innerProd_calPrecision.cpu_data(), this->local_label_->cpu_data(), this->total_label_->cpu_data(), 1);
		
		Dtype top_5_recall = GetRetrivePerformance(this_query_num, this_database_num, Caffe::RANK , this->_innerProd_calPrecision.cpu_data(), this->local_label_->cpu_data(), this->total_label_->cpu_data(), 5);
        */

		top[0]->mutable_cpu_data()[0] = loss;

        vector<int> _top_klist ;//= {1,5,10,15};
        _top_klist.push_back(1);
        _top_klist.push_back(5);
        _top_klist.push_back(10);
        _top_klist.push_back(15);

        for(int i = 1; i < top.size()-1; i++) {
            top[i]->mutable_cpu_data()[0] = GetRetrivePerformance(this_query_num, this_database_num, Caffe::RANK , this->_innerProd_calPrecision.cpu_data(), this->local_label_->cpu_data(), this->total_label_->cpu_data(), _top_klist[i - 1]);
        }

        caffe_gpu_asum(bottom[0]->count(), bottom[0]->gpu_data(), top[top.size() - 1]->mutable_cpu_data());
        top[top.size() - 1]->mutable_cpu_data()[0] /= bottom[0]->num();

        /*
		top[1]->mutable_cpu_data()[0] = top_1_recall;
		top[2]->mutable_cpu_data()[0] = top_5_recall;
        */

	}


	template <typename Dtype>
	__global__ void Get_Query_Diff_Part(const int nthreads, const int this_query_num, const int this_database_num,
		 const Dtype* innerProd_type, const Dtype* query_based_weights,
					Dtype* new_innerProd_weights){
			CUDA_KERNEL_LOOP(index, nthreads) {
				int query_idx = index / this_database_num;
				int base_idx = index % this_database_num;
                if(query_based_weights[query_idx] == 0){
                    new_innerProd_weights[index] = 0;
                }
                else{
				    new_innerProd_weights[index] = innerProd_type[index] / query_based_weights[query_idx];
                }
                /*
                if(index<4){
                    printf("index=%d, val=%f",index, new_innerProd_weights[index]);
                }
                */
			}
		}
	template <typename Dtype>
	void NPairMultiClassLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		//LOG(INFO)<<"BACKWARD STEP1 BEGIN";
		int this_query_num = this->_num;
		int this_database_num = this->_num * Caffe::NUM_GPU; 
		int fea_dim = this->_channel * this->_height * this->_width;
        int dot_normalizer = this_query_num;//1;//fea_dim

		caffe_gpu_set(this_query_num * fea_dim, (Dtype)0, this->local_feature_->mutable_gpu_diff());
		caffe_gpu_set(this_database_num * fea_dim, (Dtype)0, this->total_feature_->mutable_gpu_diff());
		//LOG(INFO)<<"BACKWARD STEP2 BEGIN";
		/*
		Dtype loss_ident_value = 0, loss_diff_value = 0;
		//下面四行的值只是求了exp后的加和部分
		caffe_gpu_mul(this_query_num * this_database_num, this->_innerProd.gpu_data(), this->_isIdentType.gpu_data(), this->_innerProd_temp1.mutable_gpu_data());
		caffe_gpu_dot(this_query_num * this_database_num, this->_innerProd_temp1.gpu_data(), this->_cross_multiplier.gpu_data(), &loss_ident_value);
		caffe_gpu_mul(this_query_num * this_database_num, this->_innerProd.gpu_data(), this->_isDiffType.gpu_data(), this->_innerProd_temp2.mutable_gpu_data());
		caffe_gpu_dot(this_query_num * this_database_num, this->_innerProd_temp2.gpu_data(), this->_cross_multiplier.gpu_data(), &loss_diff_value);
		*/
		//int fea_dim = this->_channel * this->_height * this->_width;

		if(true) { //identNum!=0 && diffNum!=0){


			Dtype loss_weight = top[0]->cpu_diff()[0];
			//这里最后加了一个内积对fea_dim的归一化,
			/*
			Dtype loss_scalar_part1 = -1 * (identNum / loss_ident_value / identNum) / fea_dim;
			Dtype loss_scalar_part2 = -1 * (-1 / ( loss_ident_value / identNum + loss_diff_value / diffNum ) / identNum) / fea_dim;
			Dtype loss_scalar_part3 = -1 * (-1 / ( loss_ident_value / identNum + loss_diff_value / diffNum ) / diffNum) / fea_dim;
			*/
			Get_Query_Diff_Part<Dtype><<<CAFFE_GET_BLOCKS(this_query_num * this_database_num), CAFFE_CUDA_NUM_THREADS>>>
				(this_query_num * this_database_num, this_query_num, this_database_num, this->_innerProd_temp1.gpu_data(),this->loss_ident_value.gpu_data(),
					this->_query_diff_part1_weights.mutable_gpu_data());
			Get_Query_Diff_Part<Dtype><<<CAFFE_GET_BLOCKS(this_query_num * this_database_num), CAFFE_CUDA_NUM_THREADS>>>
				(this_query_num * this_database_num, this_query_num, this_database_num, this->_innerProd_temp1.gpu_data(),this->_loss_value_tmp1_sum.gpu_data(),
					this->_query_diff_part2_weights.mutable_gpu_data());
			Get_Query_Diff_Part<Dtype><<<CAFFE_GET_BLOCKS(this_query_num * this_database_num), CAFFE_CUDA_NUM_THREADS>>>
				(this_query_num * this_database_num, this_query_num, this_database_num, this->_innerProd_temp2.gpu_data(),this->_loss_value_tmp1_sum.gpu_data(),
					this->_query_diff_part3_weights.mutable_gpu_data());

            /*
            for (int i=0;i<10;i++){
                LOG(INFO)<<"INDEX["<<i<<"], PART1:"<<_query_diff_part1_weights.cpu_data()[i]<<",PART2:"<<_query_diff_part2_weights.cpu_data()[i]<<",PART3:"<<_query_diff_part3_weights.cpu_data()[i];
            }
            */
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this_query_num, fea_dim, this_database_num,
				-loss_weight / dot_normalizer, this->_query_diff_part1_weights.gpu_data(), this->total_feature_->gpu_data(), (Dtype)0, this->local_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this_query_num, fea_dim, this_database_num,
				loss_weight / dot_normalizer, this->_query_diff_part2_weights.gpu_data(), this->total_feature_->gpu_data(), (Dtype)1, this->local_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this_query_num, fea_dim, this_database_num,
				loss_weight / dot_normalizer, this->_query_diff_part3_weights.gpu_data(), this->total_feature_->gpu_data(), (Dtype)1, this->local_feature_->mutable_gpu_diff());

			caffe_gpu_gemm(CblasTrans, CblasNoTrans, this_database_num, fea_dim, this_query_num,
				-loss_weight / dot_normalizer, this->_query_diff_part1_weights.gpu_data(), this->local_feature_->gpu_data(), (Dtype)0, this->total_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasTrans, CblasNoTrans, this_database_num, fea_dim, this_query_num,
				loss_weight / dot_normalizer, this->_query_diff_part2_weights.gpu_data(), this->local_feature_->gpu_data(), (Dtype)1, this->total_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasTrans, CblasNoTrans, this_database_num, fea_dim, this_query_num,
				loss_weight / dot_normalizer, this->_query_diff_part3_weights.gpu_data(), this->local_feature_->gpu_data(), (Dtype)1, this->total_feature_->mutable_gpu_diff());
			
			/*
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this_query_num, fea_dim, this_database_num,
				loss_weight * loss_scalar_part1, this->_innerProd_temp1.gpu_data(), this->total_feature_->gpu_data(), (Dtype)0, this->local_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this_query_num, fea_dim, this_database_num,
				loss_weight * loss_scalar_part2, this->_innerProd_temp1.gpu_data(), this->total_feature_->gpu_data(), (Dtype)1, this->local_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, this_query_num, fea_dim, this_database_num,
				loss_weight * loss_scalar_part3, this->_innerProd_temp2.gpu_data(), this->total_feature_->gpu_data(), (Dtype)1, this->local_feature_->mutable_gpu_diff());

			caffe_gpu_gemm(CblasTrans, CblasNoTrans, this_database_num, fea_dim, this_query_num,
				loss_weight * loss_scalar_part1, this->_innerProd_temp1.gpu_data(), this->local_feature_->gpu_data(), (Dtype)0, this->total_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasTrans, CblasNoTrans, this_database_num, fea_dim, this_query_num,
				loss_weight * loss_scalar_part2, this->_innerProd_temp1.gpu_data(), this->local_feature_->gpu_data(), (Dtype)1, this->total_feature_->mutable_gpu_diff());
			caffe_gpu_gemm(CblasTrans, CblasNoTrans, this_database_num, fea_dim, this_query_num,
				loss_weight * loss_scalar_part3, this->_innerProd_temp2.gpu_data(), this->local_feature_->gpu_data(), (Dtype)1, this->total_feature_->mutable_gpu_diff());
			*/
			//LOG(INFO)<<"BACKWARD STEP3 BEGIN";
			/*
			int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
                  */
			if (Caffe::MULTI_GPU) {
			    int count = this_database_num * fea_dim;//this->total_feature_->count();
			    //LOG(INFO) << "call mpi_allreduce in rank " << Caffe::RANK;
			    if (sizeof(Dtype) == 4) {
			        //LOG(INFO) << "start MPI_Allreduce:";
			        MPI_Allreduce(MPI_IN_PLACE, this->total_feature_->mutable_cpu_diff(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			        //LOG(INFO) << "end MPI_Allreduce";
			    } else if(sizeof(Dtype) == 8) {
			        MPI_Allreduce(MPI_IN_PLACE, this->total_feature_->mutable_cpu_diff(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			    } else {
					LOG(FATAL) << "Error size of Dtype: " << sizeof(Dtype);
			    }
			    caffe_gpu_scale(count, (Dtype)1 / Caffe::NUM_GPU, this->total_feature_->gpu_diff(), this->total_feature_->mutable_gpu_diff());
			}
			else {
				int count = this_database_num * fea_dim;
			    //LOG(INFO) << "call mpi_allreduce in rank " << Caffe::RANK;
			    if (sizeof(Dtype) == 4) {
			        //LOG(INFO) << "start MPI_Allreduce:";
			        MPI_Allreduce(MPI_IN_PLACE, this->total_feature_->mutable_cpu_diff(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			        //LOG(INFO) << "end MPI_Allreduce";
			    } else if(sizeof(Dtype) == 8) {
			        MPI_Allreduce(MPI_IN_PLACE, this->total_feature_->mutable_cpu_diff(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			    } else {
					LOG(FATAL) << "Error size of Dtype: " << sizeof(Dtype);
			    }
			    caffe_gpu_scale(count, (Dtype)1 / Caffe::NUM_GPU, this->total_feature_->gpu_diff(), this->total_feature_->mutable_gpu_diff());
			}
			//LOG(INFO)<<"BACKWARD STEP4 BEGIN";
			
			caffe_gpu_axpby(this_query_num * fea_dim,
					(Dtype)0.5,
					this->total_feature_->gpu_diff() + Caffe::RANK * this_query_num * fea_dim,
					(Dtype)0.5,
					this->local_feature_->mutable_gpu_diff()
				);
           /* 
            for(int i=0;i<10;i++)
            {
                LOG(INFO)<<i<<"diff:"<<this->local_feature_->cpu_diff()[i];
            }
            */

            /*
            //For L1 Normalization
            Dtype lambda = 0;
            caffe_gpu_sign(bottom[0]->count(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_data());
            caffe_gpu_dot(bottom[0]->count(), bottom[0]->gpu_data(), bottom[0]->gpu_diff(), &lambda);
            lambda /= -bottom[0]->count();
            caffe_gpu_axpby(bottom[0]->count(), lambda, bottom[0]->gpu_data(), (Dtype)1, bottom[0]->mutable_gpu_diff());
            for(int i=0;i<bottom[0]->count();i++){
                if(bottom[0]->cpu_diff()[i]> 1e-5){
                    LOG(INFO)<< i << ":" << bottom[0]->cpu_diff()[i];
                }
            }
            */

            
            
            

		}
			
	}

INSTANTIATE_LAYER_GPU_FUNCS(NPairMultiClassLossLayer);

}
