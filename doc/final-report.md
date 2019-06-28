## Final Report For Reproducing Paper

> **Paper title**: [Aspect Based Sentiment Analysis with Gated Convolutional Networks](https://www.aclweb.org/anthology/P18-1234>)
>
> **Authors**: Wei Xue and Tao Li
>
> **My student id and name**: 16340237	Cong Wu

### Abstract

> Abstract: problem, difficulty, idea, your key result.
>

**Aspect based sentiment analysis (ABSA)** can provide more detailed information than general sentiment analysis, because it aims to predict the sentiment polarities of the given aspects or entities in text. The authors summarize previous approaches into two subtasks: **aspect-category sentiment analysis (ACSA)** and **aspect-term sentiment analysis (ATSA)**. 

For general sentiment analysis, TextCNN proposed by Kim is a very classic CNN-based model. But for Aspect based sentiment analysis (ABSA), it seems that **no CNN-based model** has been proposed so far, most previous approaches employ **long short-term memory** and **attention mechanisms** to predict the sentiment polarity of the concerned targets, which are often complicated and need more training time. 

The authors a model based on **convolutional neural networks** and **gating mechanisms**, which is more accurate and efficient. First, the novel **Gated Tanh-ReLU Units** can selectively output the sentiment features according to the given aspect or entity. The architecture is much simpler than attention layer used in the existing models. Second, the computations of our model could be **easily parallelized** during training, because convolutional layers do not have time dependency as in LSTM layers, and gating units also work independently. 

The experiments on SemEval datasets demonstrate the efficiency and effectiveness of the author's models.

### Introduction

> Introduction: application background, research problem, related existing methods, the paper’s idea, your key results. 

Opinion mining and sentiment analysis (Pang and Lee, 2008) on user-generated reviews can provide valuable information for providers and consumers. Instead of predicting the overall sentiment polarity, fine-grained aspect based sentiment analysis (ABSA) (Liu and Zhang, 2012) is proposed to better understand reviews than traditional sentiment analysis.

Specifically, the authors are interested in the sentiment polarity of aspect categories or target entities in the text. Sometimes, it is coupled with aspect term extractions (Xue et al., 2017). 

Many existing models use LSTM layers (Hochreiter and Schmidhuber, 1997) to distill sentiment information from embedding vectors, and apply attention mechanisms (Bahdanau et al., 2014) to enforce models to focus on the text spans related to the given aspect/entity. Such models include Attention-based LSTM with Aspect Embedding (ATAE-LSTM) (Wang et al., 2016b) for ACSA; Target-Dependent Sentiment Classification (TD-LSTM) (Tang et al., 2016a), Gated Neural Networks (Zhang et al., 2016) and Recurrent Attention Memory Network (RAM) (Chen et al., 2017) for ATSA. 

However, both LSTM and attention layer are very timeconsuming during training. Certainly, it is possible to achieve higher accuracy by building more and more complicated LSTM cells and sophisticated attention mechanisms; but one has to hold more parameters in memory, get more hyper-parameters to tune and spend more time in training. 

In this paper, the authors propose a fast and effective neural network for ACSA and ATSA based on convolutions and gating mechanisms, which has much less training time than LSTM based networks, but with better accuracy. 

### Problem formulation 

> Problem formulation: formally describe the research problem. 

A number of models have been developed for ABSA, and ABSA can be divided into two subtasks, namely aspect-category sentiment analysis (ACSA) and aspect-term sentiment analysis (ATSA). 

The goal of ACSA is to predict the sentiment polarity with **regard to the given aspect, which is one of a few predefined categories**. On the other hand, the goal of ATSA is to identify the sentiment polarity **concerning the target entities that appear in the text instead, which could be**
**a multi-word phrase or a single word**. The number of distinct words contributing to aspect terms
could be more than a thousand. 

For example, in the sentence “*Average to good Thai food, but terrible delivery.*”, ATSA would ask the sentiment polarity towards the entity *Thai food*; while ACSA would ask the sentiment polarity toward the aspect *service*, even though the word *service* does not appear in the sentence. 

### Method

> Method: the basic idea, model structure. 

### Implementation

> Implementation: what you have done, difficulties & solutions



### Experiments

> Experiments: all tests, including worse and better results.



### Conclusion

> conclusion from experimental evaluation.



### Source code

> Also can find on Github: [DL-Lab-Project](<https://github.com/RQTN/DL-Lab-Project>)







