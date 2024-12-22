## Deep-ML代码笔记：深度学习

### 22.Sigmoid  Activation Function Understanding

sigmoid是常见的激活函数，其数学表达式为：
$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$
对应代码：

```python
def sigmoid(z: float) -> float:
	#Your code here
	result = 1 / (1 + math.exp(-z))
	return round(result, 4)
```

### 23.Softmax Activation Function Implementation

softmax是用于多类分类的激活函数，其数学表达式为：
$$
softmax(x_i)=\frac{e^{x_i}}{\sum_{j=1}^ne^{x_j}}
$$
其中$n$为种类数量

对应代码：

```python
def softmax(scores: list[float]) -> list[float]:
	prob_sum = sum([math.exp(score) for score in scores])
	probabilities = [round(math.exp(score) / prob_sum, 4) for score in scores]
	return probabilities
```

### 24.Single Neuron

实现单个神经元，即对于有$n$个样本、$m$个特征的输入特征矩阵$\textbf{X}\in\mathbb{R}^{n\times m}$，神经元的权重为$\textbf{w}\in \mathbb{R}^{m}$，再加上偏置$b\in \mathbb{R}^1$。其实就是对于每个样本，各个特征和权重中的各个参数一一相乘再相加。

对应核心代码：

```python
for feature in features:
		neuron_output.append(sum([f * w for f, w in zip(feature, weights)]) + bias)
```

### 25.Single Neuron with Backpropagation

对于单个神经元的反向传播，有$n$个样本、$m$个特征的输入特征矩阵$\textbf{X}\in\mathbb{R}^{n\times m}$，需要求解神经元权重梯度$\textbf{w}\in \mathbb{R}^{m}$和偏置梯度$b\in \mathbb{R}^1$并进行更新。