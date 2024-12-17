import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        #self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, n_iter=5, eta=0.01):
        stop = False
        weights = []
        weights.append(self.Wout.copy())
        errors = 0
        for i in range(n_iter):
        
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
                errors += int(eta * (target - pr) != 0.0)
            print(f"Ошибок: {errors}")      
            if errors == 0:
                stop = True
                print("Обучение сошлось.")
                break
    
            curr_weight = self.Wout.copy()
            if (any(sum(abs(curr_weight - weight)) < 1e-6 for weight in weights)):
                stop = True
                print("Алгоритм зациклился.")
                break
            
            weights.append(curr_weight)
    
    
        return self


