#### 运行截图
![cifar10](https://raw.githubusercontent.com/misads/cifar10_cnn/master/docs/cifar10.png)  

　　学习率小时收敛速度反而快

|model|learning_rate|batch|acc|
|----|----|---|---|
| vgg_a | 0.001 | 64  | 0.7467 |
| vgg_a | 0.0005 | 64  | 0.7500 |
| vgg_a | 0.0002 | 64 | 0.7517 |
| cnn | 0.001 | 64 | 0.5165 |
| cnn | 0.0005 | 64 | 0.7165 |
| cnn | 0.0002 | 64 | 0.7869 |

(cnn为3层卷积层的卷积神经网络)
