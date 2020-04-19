# Convolutional Recurrent Neural Network

This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR. For details, please refer to our paper http://arxiv.org/abs/1507.05717.

### warp-CTC installation

warp-CTC is a CTC code base of Baidu open source that can be applied to CPU and GPU efficiently and in parallel, and parallel processing of CTC algorithm.

warp-CTC installation:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
```
Add environment variables:

gedit ./.bashrc
export WARP_CTC_PATH = /home/xxx/warp-ctc/build

Example of verifying that GPU is available for warp-CTC in pytorch:

cd /home/xxx/warp-ctc/pytorch_binding/tests
python test_gpu.py

Pay attention to the main points. The installation path will be different for different Python environments. Pay attention to the installation path and determine whether you need to add environment variables! Reference blog: https://blog.csdn.net/dcrmg/java/article/details/80199722

### Generate training data
For OCR training data, you can combine [Synthetic Chinese String Dataset](https://pan.baidu.com/s/1xsuoTC711yD4s7Sp15Az0Q) extraction code: fh6h, and [ocr_data](https://github.com/CodeAchieveDream/ocr_generate_text_data) project, generate text according to your needs, train The size of the picture is usually 280 * 32, the label format is: image_path text, the training data is preferably able to combine the actual scene data, and the weaker data is identified according to the model for targeted optimization, such as numbers, English, special fonts, etc. Wait.

### train
Generally, the training data of ocr can reach ten million levels. In order to accelerate the training speed of the model, it can be converted into lmdb data for training. The steps are as follows:
1. Configure the path of the data set in tolmdb.py, and run the tolmdb.py file to generate lmdb data.
2. Set the configured data set path in crnn_train.py and run crnn_train.py for training
3. Training parameters are configured in params.py

### test
Run crnn_test.py for image testing

### Data enhancement
Perform data enhancement during training. In order to prevent overfitting during training, you can perform data enhancement in imgaug_image.py and perform data enhancement by setting different enhancers in imgaug. [Imgaug](https://imgaug.readthedocs.io/en/latest/)





