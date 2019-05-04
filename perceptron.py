import numpy as np

class Perceptron(object):
    def __init__(self, len_inputs, threshold=100, lr=0.01):
        # threshold: 훈련 시 반복 횟수
        # lr: 훈련계수(learning rate)
        self.threshold = threshold
        self.lr = lr
        # (bias +) 가중치 벡터 초기화
        self.weights = np.zeros(len_inputs + 1)

    def predict(self, inputs):
        # h(x)= dot(x, w) + bias(=-theta)
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # 활성화 함수 사용
        # pdf p.11
        if sum > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        # threshold 만큼 반복
        for _ in range(self.threshold):
            # training_input, labels에서 반복
            # pdf p.28
            for inputs, label in zip(training_inputs, labels):
                # 현재 input에 대한 예측 값 확인
                prediction = self.predict(inputs)
                # 가중치 벡터(w) 업데이트
                # w = w + lr*yi*xi
                self.weights[1:] += self.lr * (label - prediction) * inputs
                # b = b + lr*yi
                self.weights[0] += self.lr * (label - prediction)
