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

==**my key results**==: haven't written yet. The key results will choose from [Experiments part](#**Experiments**) (Acc and Time to converge).

### Problem formulation 

> Problem formulation: formally describe the research problem. 

A number of models have been developed for ABSA, and ABSA can be divided into two subtasks, namely aspect-category sentiment analysis (ACSA) and aspect-term sentiment analysis (ATSA). 

The goal of ACSA is to predict the sentiment polarity with **regard to the given aspect, which is one of a few predefined categories**. On the other hand, the goal of ATSA is to identify the sentiment polarity **concerning the target entities that appear in the text instead, which could be**
**a multi-word phrase or a single word**. The number of distinct words contributing to aspect terms
could be more than a thousand. 

For example, in the sentence “*Average to good Thai food, but terrible delivery.*”, ATSA would ask the sentiment polarity towards the entity *Thai food*; while ACSA would ask the sentiment polarity toward the aspect *service*, even though the word *service* does not appear in the sentence. 

### **Method**

> Method: the basic idea, model structure. 

The authors' model GCAE can handle both ACSA task and ATSA task well, but its architecture is slightly different between the two tasks, mainly in the **embedding of the aspect information**. 

For ACSA task, **GCAE mainly consists of an embedding layer, a pair of one-dimension convolutional layer, GTRU gate and a max-pooling layer.** Figure 1 illustrates the authors’ model architecture for ACSA task.

<center><img src='img/GCAE-ACSA.png' height=400px><center>
<center><span>Figure 1: Illustration of the authors' model GCAE for ACSA task</span></center>

Suppose we now have a sentence $X$, $X$ contains $L$ words, through the embedding layer, each word can be converted into a $D$-dimension word vector, at this time, the sentence $X$ is represented as a $D\times L$ matrix.
$$
\begin{align}
X&=[v_1, v_2, \cdots, v_L]\tag{1}
\end{align}
$$
The **one-dimension** convolutional layer convolves the embedding vectors input $X$ with multiple convolutional kernels of different widths. We use the filter $W_c \in R^{D\times k}(k<L)$ to scan on the sentence matrix X. For each scan position, we can get a convolution result $X_{i:i+k}*W_c$.

**A pair of convolutional neuron computes features for a pair of gates: tanh gate and ReLU gate. The ReLU gate receives the given aspect information.** The outputs of two gates are element-wisely multiplied, thus, we compute the feature $c_i$ as:
$$
\begin{align}
a_i&=relu(X_{i:i+k}*W_a+V_av_a+b_a)\tag{2}\\\\
s_i&=tanh(X_{i:i+k}*W_s+b_s)\tag{3}\\\\
c_i&=s_i\times a_i\tag{4}
\end{align}
$$
where $v_a$ is the embedding vector of the given aspect category in ACSA.

We can set $n_k$ filters of the same width k, the output features of a sentence then form a matrix $C\in R^{n_k\times L_k}$. For each convolutional filter, the max-over-time pooling layer takes the maximal value among the generated features, i.e., the maximal value of each line of $C$, resulting in a fixed-size vector $e\in R^{n_k}$.

Finally, the final fully-connected layer with softmax function uses the vector $e$ to predict the sentiment polarity $\hat{y}$. The model is trained by minimizing the cross-entropy loss between the ground-truth $y$ and the predicted value $\hat{y}$ for all data samples.

For ATSA task, the given aspect information consists of multiple words instead of one word in ACSA task, **the authors simply extend GCAE by adding a small convolutional layer on aspect terms** to deal with this situation, as shown in Figure 2.

<center><img src='img/GCAE-ATSA.png'><center>
<center><span>Figure 2: Illustration of the authors' model GCAE for ATSA task. It has an additional convolutional layer on aspect terms.</span></center>

In ACSA, the aspect information controlling the flow of the sentiment features in GTRU is from one aspect word; while in ATSA, such information is provided by a small CNN on aspect terms $[w_i, w_{i+1}, \cdots, w_{i+k}]$. The additional CNN extracts the important features from multiple words while retains the ability of parallel computing.

### **Implementation**

> Implementation: what you have done, difficulties & solutions



### **Experiments**

> Experiments: all tests, including worse and better results.

- ACSA Accuracy: GCAE vs. ATAE-LSTM
  - Accuracy on Semval ACSA Restaurant-Large dataset (normal version and hard version)

- ATSA Accuracy: GCAE vs. TD-LSTM
  - Accuracy on Semval ATSA Restaurant dataset (normal version and hard version)
- The time to converge in seconds on ATSA task: GCAE, ATAE-LSTM, TD-LSTM
- The accuracy of different gating units on restaurant reviews on ACSA task: GTU, GLU, GTRU

### Conclusion

> Conclusion: conclusion from experimental evaluation.



### **Source code**

> Also can find on Github: [DL-Lab-Project](<https://github.com/RQTN/DL-Lab-Project>)







