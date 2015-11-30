#include <algorithm>
#include <vector>
#include <algorithm>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  // CHECK_EQ(bottom[2]->channels(), 1);
  // CHECK_EQ(bottom[2]->height(), 1);
  // CHECK_EQ(bottom[2]->width(), 1);
  diff_a_p_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_a_n_.Reshape(bottom[1]->num(), bottom[1]->channels(), 1, 1);
  dist_sq_a_p_.Reshape(bottom[0]->num(), 1, 1, 1);
  dist_sq_a_n_.Reshape(bottom[0]->num(), 1, 1, 1);
  // gpu
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
    count, 
    bottom[0]->cpu_data(), // anchor
    bottom[1]->cpu_data(), // positive
    diff_a_p_.mutable_cpu_data()); // anchor_i-positive_i
  caffe_sub(
    count, 
    bottom[0]->cpu_data(), // anchor
    bottom[2]->cpu_data(), // negative
    diff_a_n_.mutable_cpu_data()); // anchor_i-negative_i

  const int channels = bottom[0]->channels();
  Dtype margin = 1.0; //this->layer_param_.triplet_loss_param().margin();
  
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_a_p_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_a_p_.cpu_data() + (i*channels), diff_a_p_.cpu_data() + (i*channels));
    dist_sq_a_n_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_a_n_.cpu_data() + (i*channels), diff_a_n_.cpu_data() + (i*channels));
    loss += std::max<Dtype>(pow(sqrt(dist_sq_a_p_.cpu_data()[i]), 2) 
      - pow(sqrt(dist_sq_a_n_.cpu_data()[i]), 2) + margin, Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = 1.0; //this->layer_param_.triplet_loss_param().margin();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        Dtype mdist(0.0);
        Dtype beta(0.0);
        Dtype dist = pow(sqrt(dist_sq_a_p_.cpu_data()[j]), 2) 
            - pow(sqrt(dist_sq_a_n_.cpu_data()[j]), 2);
        mdist = dist + margin;
        beta = -alpha * mdist / (dist + Dtype(1e-4));
        if (mdist > Dtype(0.0)) {
          caffe_cpu_axpby(
            channels,
            beta,
            diff_a_p_.cpu_data() + (j*channels), // TODO: cannot work
            Dtype(0.0),
            bout + (j*channels));
        } else {
          caffe_set(channels, Dtype(0), bout + (j*channels));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
