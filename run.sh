cd /home/ubuntu/personality-detection/

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
PATH=${CUDA_HOME}/bin:${PATH}

export PATH

mkdir t0w2v
python conv_net_train_gpu.py -static -word2vec 0
mv cvt* t0w2v
mv perf_output_* t0w2v

mkdir t1w2v
python conv_net_train_gpu.py -static -word2vec 1
mv cvt* t1w2v
mv perf_output_* t1w2v

mkdir t2w2v
python conv_net_train_gpu.py -static -word2vec 2
mv cvt* t2w2v
mv perf_output_* t2w2v

mkdir t3w2v
python conv_net_train_gpu.py -static -word2vec 3
mv cvt* t3w2v
mv perf_output_* t3w2v


mkdir t4w2v
python conv_net_train_gpu.py -static -word2vec 4
mv cvt* t4w2v
mv perf_output_* t4w2v
