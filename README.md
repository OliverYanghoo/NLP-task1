# NLP-task1

## 一、 提交说明

提交内容：实验报告（PDF格式）、完整代码、代码运行说明。

提交方式：包含所有文件的 GitHub 链接。

截止日期：10月12日上午 11:55。

独立完成：本次作业要求独立完成。

## 二、 作业目标

旨在帮助理解和掌握不同文本表示方法在文本分类任务中的基本原理与实际效果。具体目标包括：

理解不同文本表示方法的基本思想；
掌握文本分类任务的基本实验流程；
熟悉 Logistic Regression 在文本分类中的使用；
掌握 Word2Vec、GloVe 等词向量的基本使用方法；
熟悉 BERT 类预训练语言模型的 Fine-tuning 流程；
能够使用 Accuracy 和 Macro-F1 对文本分类模型进行评价；
能够对不同文本表示方法的性能进行基本比较和分析。

## 三、 数据集

New York Times (NYT) 新闻数据集：主要分类数据集，包含 text（新闻文本）和 label（新闻类别）两列。

AG News 数据集：提供新闻文本，主要用于 Task 2 中训练 Word2Vec 词向量。

## 四、 数据划分与评价指标

NYT 数据集划分：需先随机打乱，统一按以下比例划分，且所有实验应使用相同划分以保证可比性：

Training Set: 80%
Validation Set: 10%
Test Set: 10%

评价集：最终模型性能统一在 NYT Test Set 上评价。

评价指标：

Accuracy：分类正确样本数 / 总测试样本数。
Macro-F1：各类别 F1 Score 的平均值，用于全面反映模型在不均衡类别上的能力。
评分说明：评分关注方法和实现的正确性，合理范围内的随机性能波动不影响成绩。

## 五、 具体任务要求

### Task 1：Bag of Words（30分）

统一使用 Logistic Regression 作为分类器，在 NYT Test Set 上报告 Accuracy 和 Macro-F1。需完成以下三种表示方法：

- Binary Bag of Words：构建词表，文档表示为 |V| 维向量。若词表中第 i 个单词出现在文档中则 x_i = 1，否则为 0（仅考虑是否出现，不考虑频次）。分词可使用 nltk.word_tokenize。
- Word Frequency：记录每个单词在当前文档中的出现次数 c_i，文档向量为 (c_1, c_2, ..., c_|V|)。

注：原文提及“三种”方法，但详细描述仅列出上述两种，请以实际文档或课程补充为准。

### Task 2：Word2Vec（50分）

统一使用 100维词向量，分类器使用 Logistic Regression。需完成以下三组实验并报告指标：

- Pre-trained GloVe：使用公开的 glove.6B 100维词向量。文档向量通过所有有效单词词向量的平均值得到：e(d) = 1/n * Σ e(w_i)。
- Word2Vec Trained on AG News：使用 AG News 数据自行训练 100维 Word2Vec 模型（可用 gensim），再对 NYT 文档计算平均词向量进行分类。
- Word2Vec Trained on NYT：使用 NYT 数据自行训练 100维 Word2Vec 模型，再计算文档表示并进行分类。

### Task 3：Pre-trained Neural Models（20分）

使用预训练语言模型微调完成分类，在 NYT Test Set 上报告 Accuracy 和 Macro-F1。

BERT Fine-tuning：

- 模型：google-bert/bert-base-uncased
- Max length：64
- Epochs：3
