# Not-a-Diagnosis
## 简介
Not-a-Diagnosis 是一个基于3DCNN和ResNet的医学影像分类模型。该模型基于 LIDC-IDRI 数据集的子集进行训练，子集是经过数据预处理截取的，$\mathbb{R}^{64\times64\times64}$空间中的 Patch，并遵循大部分文献选用的方式将该数据集中 label 为1和2的样本划为 benign（良性）, label 为4和5的样本划为 malignant（恶性）。

具体来说，该模型完成的是对肺结节的CT影像数据进行的分类，模型输入是一个包含了肺结节的 Patch， 输出是判定为 malignant 的概率，理论上来讲这个概率越大，则肺结节癌变的可能性就越高。

在测试过程中，最优的性能记录如下：
```
Accuracy: 0.9771986970684039, 
AUC: 0.9791390112275913,
Confusion matrix:
[[682   4]
 [ 17 218]]
```
因为这个子集是我自己划分出来的，所以这个性能仅供参考。

## 来源
这是我在山东大学读本科时，认知科学与类脑计算的课设，课设选题里有一条“医学影像分割与分类”

其实最开始我是想做肺部孢子菌感染的CT影像识别 or 分类的，但是不幸的是这一类病症因为菌种多样性太多，很难找到一个统一的分类标准，所以没有人做出可用的数据集。

至于之前那个选题的原因——我姥爷是今年夏天走的，简而言之，是因为肺癌晚期做化疗和放疗，不过我感觉主要是因为放疗，导致免疫功能严重受损，我估计当时他的胸腺功能已经破坏的不成样了，而又因此不幸肺部感染了，前后可能也就一个月的时间，肺部就几乎被菌丝布满了，这导致他的肺几乎不可用了，简单粗暴的说，被孢子菌憋死了。医生说孢子菌感染对于健康人来说是没什么影响的，即使吸入了也会被免疫系统消灭掉，但对于他来说就是致命的了。

总之，找不到肺部孢子菌感染的数据集的理由有很多，所以最终我决定退而求其次，转向对肺部恶性肿瘤的识别。

这项目我也完全开源了，至少从功能设计上来看，或者你从名称上也能看出来，这不是给你诊断结果，就只是做个参考，真实情况还是得看医生，要是信得过我的话可以用用，不过如果你真有需求还是建议去看看我参考文献里那些大佬们做的成果。

## 使用方式
TODO
## 参考文献
```
[1] L. E. Chetan et al., “Timely Detection of Lung Nodule Malignancy Using 3D Convolutional Neural Networks,” Cureus Journals, vol. 2, no. 1, July 2025, doi: 10.7759/s44389-025-06619-1.
[2] M. Kashyap et al., “Automated Deep Learning-Based Detection and Segmentation of Lung Tumors at CT,” Radiology, vol. 314, no. 1, p. e233029, Jan. 2025, doi: 10.1148/radiol.233029.
[3] D. A. E.-S. Mansour, “Automated Pulmonary Nodule Detection in LDCT Using 3D ResNet and Adaptive Patch Strategy,” 2025.
[4] J. Ning, H. Zhao, L. Lan, P. Sun, and Y. Feng, “A Computer-Aided Detection System for the Detection of Lung Nodules Based on 3D-ResNet,” Applied Sciences, vol. 9, no. 24, p. 5544, Jan. 2019, doi: 10.3390/app9245544.
[5] Sakshiwala and M. P. Singh, “Channel attention-based 3D CNN for classification of pulmonary nodules,” in 8th International Conference on Computing in Engineering and Technology (ICCET 2023), July 2023, pp. 532–536. doi: 10.1049/icp.2023.1544.
[6] W. Shen et al., “Multi-crop Convolutional Neural Networks for lung nodule malignancy suspiciousness classification,” Pattern Recognition, vol. 61, pp. 663–673, Jan. 2017, doi: 10.1016/j.patcog.2016.05.029.
[7] S. G. Armato et al., “The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans,” Medical physics, vol. 38, no. 2, pp. 915–31, 2011, doi: 10.1118/1.3528204.
```
