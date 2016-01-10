#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
	CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

  	 // Configure the kernel size, padding, stride, and inputs.
  LocallyConnectedParameter local_param = this->layer_param_.locally_connected_param();
 	
	num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

	kernel_size_ = local_param.kernel_size();
	stride_ = local_param.stride();
  pad_ = local_param.pad();
  bias_term_ = local_param.bias_term();
  num_output_ = local_param.num_output();

  height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  width_out_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

  M_ = num_output_;
  K_ = channels_ * kernel_size_ * kernel_size_;
  N_ = height_out_ * width_out_;

  CHECK_GT(num_output_, 0);
  CHECK_GE(height_, kernel_size_) << "input height smaller than kernel size";
	CHECK_GE(width_, kernel_size_) << "input width smaller than kernel size";

	//Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skiping parameter initialization";
	} else {
		if (bias_term_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		//Initialize the weights
		this->blobs_[0].reset(new Blob<Dtype>(
			num_output_, 1, K_, N_));
		// fill the weights
    	shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(local_param.weight_filler()));
    	weight_filler->Fill(this->blobs_[0].get());
    	// If necessary, intiialize and fill the bias term
    	if (bias_term_) {
      		this->blobs_[1].reset(new Blob<Dtype>(1, 1, M_, N_));
      		shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(local_param.bias_filler()));
      		bias_filler->Fill(this->blobs_[1].get());
      	}
	}
}


template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

	CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with weights.";

  	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    	CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    	CHECK_EQ(channels_, bottom[bottom_id]->channels())
      		<< "Inputs must have same channels.";
    	CHECK_EQ(height_, bottom[bottom_id]->height())
      		<< "Inputs must have same height.";
    	CHECK_EQ(width_, bottom[bottom_id]->width())
      		<< "Inputs must have same width.";
  	}

  	// Shape the tops.
  	for (int top_id = 0; top_id < top.size(); ++top_id) {
    	top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
 	}

  	// The im2col result buffer would only hold one image at a time to avoid
  	// overly large memory usage.
  	col_buffer_.Reshape(
      	1, channels_ * kernel_size_ * kernel_size_, height_out_, width_out_);

  	for (int top_id = 0; top_id < top.size(); ++top_id) {
    	top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  	}

}

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	
	const Dtype* weights = this->blobs_[0]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	Dtype* col_data = col_buffer_.mutable_cpu_data();

	//a vector of ones
	Blob<Dtype> ones;
  	ones.Reshape(1, 1, 1, K_);
  	FillerParameter filler_param;
  	filler_param.set_value(1);
  	ConstantFiller<Dtype> filler(filler_param);
  	filler.Fill(&ones);	

  	Blob<Dtype> middle_result;
  	middle_result.Reshape(1, 1, K_, N_);

	for (int n = 0; n < num_; ++n) {
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
			width_, kernel_size_, kernel_size_,
			pad_, pad_, stride_, stride_, col_data);
		for (int m = 0; m < num_output_; ++m)	{
			caffe_mul(K_ * N_, col_data, weights + this->blobs_[0]->offset(m),
				middle_result.mutable_cpu_data());
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_, 
				1., ones.cpu_data(), middle_result.mutable_cpu_data(),
				0., top_data + top[0]->offset(n, m));
		}
		
		if (bias_term_) {
			caffe_add(M_ * N_, this->blobs_[1]->cpu_data(),
				top_data + top[0]->offset(n), top_data + top[0]->offset(n));
		}

	}



}

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){


	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* col_data = col_buffer_.mutable_cpu_data();
	Dtype* col_diff = col_buffer_.mutable_cpu_diff();
	const Dtype* weights = this->blobs_[0]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = NULL;
	Blob<Dtype> middle_result;
	middle_result.Reshape(1, 1, 1, N_);

	Blob<Dtype> xt;
  	xt.Reshape(1, 1, K_, N_);
  	Dtype* xt_data = xt.mutable_cpu_data();

  	if (bias_term_) {
   		bias_diff = this->blobs_[1]->mutable_cpu_diff();
    	caffe_set(this->blobs_[1]->count(), Dtype(0.0), bias_diff);
    	for (int n = 0; n < num_; ++n) {
      		caffe_add(M_ * N_, bias_diff,
          	top_diff + top[0]->offset(n),
          	bias_diff);
    	}
  	}

  	caffe_set(this->blobs_[0]->count(), Dtype(0.0), weight_diff);
  	for (int n = 0; n < num_; n++) {
    	im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        	width_, kernel_size_, kernel_size_,
        	pad_, pad_, stride_, stride_, col_data);

    	// gradient w.r.t. weight
    	for (int m = 0; m < num_output_; m++) {
      		Dtype* filter_weight_diff = weight_diff+this->blobs_[0]->offset(m);
      		for (int k = 0; k < K_; k++) {
        		caffe_mul(N_, top_diff+top[0]->offset(n, m),
            	col_data + col_buffer_.offset(0, k), xt_data+xt.offset(0, 0, k));
      		}
      	caffe_cpu_axpby(K_*N_, Dtype(1.0), xt_data,
          Dtype(1.0), filter_weight_diff);
    	}

    	// gradient w.r.t. bottom data
    	if (propagate_down[0]) {
      		caffe_set(col_buffer_.count(), Dtype(0.0), col_diff);
      	for (int m = 0; m < num_output_; m++) {
        	for (int k = 0; k < K_; k++) {
          		caffe_mul(N_, top_diff+top[0]->offset(n, m),
              		weights+this->blobs_[0]->offset(m, 0, k),
              		middle_result.mutable_cpu_data());

          		caffe_cpu_axpby(N_, Dtype(1.0),
              		middle_result.cpu_data(), Dtype(1.0),
              		col_diff + col_buffer_.offset(0, k));
        	}
      	}

      	// col2im back to the data
      	col2im_cpu(col_diff, channels_, height_,
          	width_, kernel_size_, kernel_size_,
          	pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));
    	}
  	}
}







#ifdef CPU_ONLY
STUB_GPU(LocallyConnectedLayer);
#endif

INSTANTIATE_CLASS(LocallyConnectedLayer);
REGISTER_LAYER_CLASS(LocallyConnected);


}  // namespace caffe
