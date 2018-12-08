# Reference

This project is based on 

https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation 

Please visit this link first, where you can find the corresponding papar, code and introduction.


# Download

链接：https://pan.baidu.com/s/1NizpK57fAYjlGuMKG-cXPw 
提取码：v3ik 
复制这段内容后打开百度网盘手机App，操作更方便哦

Here you can download the dataset and some good-performing models

PascalVocCoseg.zip : the dataset

epoch1iter5000.pkl : the author's CA model with batch size 4

CAepoch0iter5000.pkl : my CA model with batch size 16

FCAepoch0iter7000.pkl : my FCA model with batch size 16

CSAepoch0iter6000.pkl : my CSA model with batch size 16

vggepoch0iter2000.pkl : my vgg19 model with batch size 16


# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- pytorch = 0.4
- Pillow
- Numpy

# Train

Unzip the dataset into Datasets folder with name **PascalVoc**.

Then use:

``` 
python train.py 
```

You can specify the address by yourself:

```
python train.py --train_data "Datasets/PascalVoc/image/" --train_label "Datasets/PascalVoc/colabel/train/" --train_txt "Datasets/PascalVoc/colabel/train.txt" --val_data "Datasets/PascalVoc/image/" --val_label "Datasets/PascalVoc/colabel/val/" --val_txt "Datasets/PascalVoc/colabel/val.txt" --model_path "model_path/"
```


**NOTICE**

The default batch_size is 16, and it may consume a large amount of memory

If there is an out-of-memory error, use this:
```
python train.py --batch_size=4
```


# Test

There are some good-performing models at the baidu cloud link, you can download them and put them in the model_path folder.

Or you can train a model by yourself and save it in the model_path folder.

Then you can co-segment two images like this:

```
python single_demo.py --image1 "demo/1.jpg" --image2 "demo/2.jpg" --output1 "demo/demo_1.jpg"  --output2 "demo/demo_2.jpg" --model "model_path/epoch1iter5000.pkl"
```

Or co-segment many images like this:

```
python group_demo.py --image_path "group_demo/images/" --output_path "group_demo/demo_outputs/" --model "model_path/epoch1iter5000.pkl"
```



# Evaluate

To evaluate a model, you can do this:

```
python evaluation.py --filename=model_path/CAepoch0iter5000.pkl
```

# Models

By default train.py and evaluation.py use CA model structure 
, that is, a model with CA attention learner implemented by 
models/model_ca.py,

If you want to train or evaluate a modle with a different structure, you need 
to specify the model structure with parameter `--model=`.

For example, to train a CSA model you need to use:
```
python train.py --model=csa
```

and to evaluate a CSA model you need to use: 
```
python evaluation.py --model=csa --filename=model_path/CSAepoch0iter6000.pkl
```

I've implemented as follow model structures: 

ca : a standard model with CA attention learner

fca : a standard model with FCA attention learner

csa : a standard model with CSA attention learner

fcsa : a standard model with FCSA attention learner, this learner is a
combination of FCA and CSA

blank : a blank contrast, no attention learner

self : another blank contrast, use the attention learner on itself

inception : replace the vgg16 encoder with a inception_v3 network

vgg : replace the vgg16 encoder with a vgg-19 network

vggbn : replace the vgg16 encoder with a vgg-19 network enhanced by batch 
normalization layers
  
resnet : replace the vgg16 encoder with a resnet152 network
   
resnet_improved : replace the vgg16 encoder with a resnet152 network and 
extend the attention learner to 2048 channels

**NOTICE**

most of them fail.


# Result

I've already placed some demo results in this project.

In the demo folder, my_1.jpg and my_2.jpg are generated by my model.

In the group_demo folder, my_outputs are generated by my model.

You can find that they are slightly different from the author's results.

In the logs folder, there are logs containing time consumption
and evaluation information. 