# 这是一个利用ConvNeXt_base模型进行星系分类的课题
## 第一步先处理数据
 1. 首先你需要先下载数据集https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data
 2. 解压后对应修改Tree_wash.py脚本文件中申明的地址（Tree_wash中还包含一个显示函数可以直观的看到数据增强的样本）
 3. 随后执行脚本中决策树分类函数（5个类别参考决策树的阈值）和复制图片函数
 > 注：本次课题的数据样本清洗参考了戴加明的论文(https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201901&filename=1018126678.nh&uniplatform=NZKPT)如果你想自己设置阈值划分类别请参考原文(https://arxiv.org/pdf/1308.3496.pdf)
 > 先叠个甲Tree_wash.py脚本是早期写的,由于仅供一次性使用没有注重代码效率和优化。如果您看到代码后有感到不适，我很抱歉。but!（如果可以感谢帮忙优化Q^Q）
## 第二步开始训练
- （可选）先下载预训练模型（可以在model.py文件找到对应模型的预训练模型文件下载连接）
- 建议详细阅读utils.py文件和model.py。utils.py中进行了数据的预处理操作，model.py给出了官方的ConvNeXt详细代码并学习其中的设计思路
- 修改train.py中的对应地址既可执行训练。可以在model.py文件查看ConvNeXt网络的结果代码实现，以及获得对应的预训练集连接。本课题采用base_1k_224模型兼顾网络性能和体积
- 训练结束后将会得到Test_name.csv文件包含所有测试集的文件名和对应标签，以便于预测和测试使用。
## 第三步模型评估
- 文件夹model中包含一个在base预训练模型基础上训练得到的“最优”模型（准确率达到96.9%）您可以直接使用它进行预测，评估或者再次训练（可能会过拟合）。
- 利用predick.py脚本对train.py文件中所有的测试集样本进行测试（可以自己选择测试用的模型），可获得所有测试集图片的详细信息（预测类别，各类别置信度等）all_output.csv（predick.py中还包含了显示函数，可以输出随机图片的预测结果。）
- 随后可以用model_evaluation.ipynb进行模型评估，PR曲线，ROC曲线，混淆矩阵等。并获得《各类别准确率评估指标.csv》
- 建议详细阅读predick.py和model_evaluation.ipynb学习其中的设计思路

### 致谢及说明
- 如果你想了解深度学习或只是ConvNeXt网络强烈建议观看b站up“霹雳吧啦Wz”的视频(【13.1 ConvNeXt网络讲解】 https://www.bilibili.com/video/BV1SS4y157fu/?share_source=copy_web&vd_source=63c41f55b275e7810655017c0e3e9863)
