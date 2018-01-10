#ifndef CAFFE_NPAIR_MULTI_CLASS_LOSS_LAYER_HPP_
#define CAFFE_NPAIR_MULTI_CLASS_LOSS_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/confusion_matrix.hpp"

namespace caffe {
  //Single machine multi-gpu card
  template<typename Dtype>
  class NPairMultiClassLossLayer : public LossLayer<Dtype>
  {
  public:
    explicit NPairMultiClassLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {
        
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "NPairMultiClassLoss"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }//features labels
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }//Loss [, top-1 retrieve acc, top-5 retrieve acc]
    virtual inline int MaxTopBlobs() const { return 5; }
    ~NPairMultiClassLossLayer(){}
    static bool comp(Dtype a, Dtype b) {
      return a > b;
    }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    /*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/
  private:

    void GatherFeatureAndLabel();
    Dtype GetRetrivePerformance(const int query_num, const int global_num, const int query_begin_idx_in_global, const Dtype* distanceMtx, 
      const Dtype* query_labels, const Dtype* global_labels, const int top_k);
    //Dtype _theata;//threta is saved in blobs_[0]
    int _num, _channel, _width, _height;
    
    Blob<Dtype> _innerProd, _innerProd_temp1, _innerProd_temp2, _cross_multiplier,
         _isIdentType, _isDiffType, _isSelectPair, _tmp_Select_Ident, _tmp_Select_Diff;
    Blob<Dtype> _innerProd_calPrecision;
    Blob<Dtype> _query_multiplier;

    Blob<Dtype> *local_feature_, *total_feature_;
    Blob<Dtype> *local_label_, *total_label_;

    Blob<Dtype> identNum, diffNum, loss_ident_value, loss_diff_value , _loss_value_tmp1_sum, _loss_value_tmp2_div, _loss_value_tmp3_log;
    //Dtype loss_ident_value, loss_diff_value;

    Blob<Dtype> _min_within_dot_blob, _max_between_dot_blob, _max_all_dot_blob;
    Blob<Dtype> _nega_select_threshold, _posi_select_threshold;

    Blob<Dtype> loss_scalar_part1, loss_scalar_part2, loss_scalar_part3;

    Blob<Dtype> _query_diff_part1_weights, _query_diff_part2_weights, _query_diff_part3_weights;
    Dtype _margin_diff, _margin_ident, _identSN, _diffSN;
  };

}  // namespace caffe

#endif  // CAFFE_NPAIR_MULTI_CLASS_LOSS_LAYER_HPP_
