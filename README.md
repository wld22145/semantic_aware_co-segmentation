NOTICE
This project is unfinished, so it has only partial functions.
This project is only used by me, so it has very messy layout.

This project is based on https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation 
There are corresponding paper, codes and instructions.

1.Install:

follow the instruction at the github link.

NOTICE
actually the code requires more packages. 
Some packages are never used. 
But I still keep those import code.

2.Train:

to train a model you need to :
download the dataset from the github link.
unzip the dataset into Datasets folder with name PascalVoc.
follow the instruction at the link.
Because I modified the defualt address so directly use " python train.py " can also work.
Unless the defualt address doesn't suit you.
NOTICE
I set the defualt batch_size as 16, and it may consume 32G memory
if there is out-of-memory error, set the batch_szie to 4


3.Use:

to use a model you need to:
there must be a model existing in the model_path folder
if there isn't, then train one or download one
you can download the author's model from the github link, or mine from 

链接：https://pan.baidu.com/s/1Nm9JC5Q5FQ9G1XrZ6zjWUQ 
提取码：5io9 
复制这段内容后打开百度网盘手机App，操作更方便哦

then follow the instruction at the link


4.Evaluate:

to evaluate a model with 3 metrics you need to:

NOTICE
This part is mixed in train.py and very messy now


you need to modify the end of train.py
trainer = Trainer('train')
to
trainer = Trainer('test')

and input a filename the test() function

I will clean it up soon

5.Results:

in the demo folder, I already placed two extra images generated by my model.
in the group_demo folder, I already placed an extra folder of images generated by my model.
in the logs model, I placed many log information files generated when training. There are time consumption and evaluation information. But They are not very uniform.

6.Models

By defualt the train.py uses model_ca.py, that is, a model with CA attention learner.
I implement CSA, FCA too.
To a train model with them you need to:
modify the train.py
set 
self.mode
to a proper name, which determines your model's name when you save it

set 
self.net = model_ca.model().cuda()
to
self.net = model_csa.model().cuda()
or something like it to use CSA or other models

I implement FCSA, which is a new fusion method I created.
I implement a new model structure based on inception_v3 model but it fails completely. This model is at model_inception_fail.py



