# Deep Learning-Based Document Modeling for Personality Detection from Text

This code implements the model discussed in [Deep Learning-Based Document Modeling for Personality Detection from Text](http://sentic.net/deep-learning-based-personality-detection.pdf) for detection of Big-Five personality traits, namely:

-   Extroversion
-   Neuroticism
-   Agreeableness
-   Conscientiousness
-   Openness


## Requirements

-   Ubuntu 16.0.4 64bit (Tested)
-   Python 2.7
-   Theano 1.0.4 (Tested)
-   Pandas 0.24.2 (Tested)
-   Pre-trained [GoogleNews word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) vector (If you are using ssh try [this](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz))


## Preprocessing

`process_data.py` prepares the data for training. It requires three command-line arguments:

1.  Path to [google word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) file (`GoogleNews-vectors-negative300.bin`)
2.  Path to `essays.csv` file containing the annotated dataset
3.  Path to `mairesse.csv` containing [Mairesse features](http://farm2.user.srcf.net/research/personality/recognizer.html) for each sample/essay

This code generates a pickle file `essays_mairesse.p`.

Example:

```sh
python process_data.py ./GoogleNews-vectors-negative300.bin ./essays.csv ./mairesse.csv
```

## Configuration for training the model

A. **Running using CPU**
1. Configure ~./theanorc:
```sh
[global]
floatX=float64
OMP_NUM_THREADS=20
openmp=True
```

B. **Running using GPU**
1. Install [libgpuarray](http://deeplearning.net/software/libgpuarray/installation.html)
2. Install [cuDNN](http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html) for faster training
3. Add CUDA path to .bashrc:
```sh
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
PATH=${CUDA_HOME}/bin:${PATH}

export PATH
```
4. Configure ~/.theanorc:
```sh
[cuda]
root=/usr/local/cuda
[global]
device=cuda
floatX = float32
OMP_NUM_THREADS=20
openmp=True

[nvcc]
fastmath=True
```

## Training

Note: Before these changes, every epoch took about 5 hours to complete. After them, it took less than an hour on CPU and about 45s on GPU (Improvements depend on your system spec)

A. **Running on GPU**

`conv_net_train_gpu.py` trains and tests the model using GPU.(Alternatively, you can run "run.sh" and train all traits using word2vec at once)

B. **Running on CPU**

`conv_net_train.py` trains and tests the model using CPU.

Both scripts require three command-line arguments:

1.  **Mode:**
    -   `-static`: word embeddings will remain fixed
    -   `-nonstatic`: word embeddings will be trained
2.  **Word Embedding Type:**
    -   `-rand`: randomized word embedding (dimension is 300 by default; is hardcoded; can be changed by modifying default value of `k` in line 111 of `process_data.py`)
    -   `-word2vec`: 300 dimensional google pre-trained word embeddings
3.  **Personality Trait:**
    -   `0`: Extroversion
    -   `1`: Neuroticism
    -   `2`: Agreeableness
    -   `3`: Conscientiousness
    -   `4`: Openness

Example:

```sh
python conv_net_train.py -static -word2vec 2
```


## Citation

If you use this code in your work then please cite the paper - [Deep Learning-Based Document Modeling for Personality Detection from Text](http://sentic.net/deep-learning-based-personality-detection.pdf) with the following:

```
@ARTICLE{7887639, 
 author={N. Majumder and S. Poria and A. Gelbukh and E. Cambria}, 
 journal={IEEE Intelligent Systems}, 
 title={{Deep} Learning-Based Document Modeling for Personality Detection from Text}, 
 year={2017}, 
 volume={32}, 
 number={2}, 
 pages={74-79}, 
 keywords={feedforward neural nets;information filtering;learning (artificial intelligence);pattern classification;text analysis;Big Five traits;author personality type;author psychological profile;binary classifier training;deep convolutional neural network;deep learning based method;deep learning-based document modeling;document vector;document-level Mairesse features;emotionally neutral input sentence filtering;identical architecture;personality detection;text;Artificial intelligence;Computational modeling;Emotion recognition;Feature extraction;Neural networks;Pragmatics;Semantics;artificial intelligence;convolutional neural network;distributional semantics;intelligent systems;natural language processing;neural-based document modeling;personality}, 
 doi={10.1109/MIS.2017.23}, 
 ISSN={1541-1672}, 
 month={Mar},}
```
