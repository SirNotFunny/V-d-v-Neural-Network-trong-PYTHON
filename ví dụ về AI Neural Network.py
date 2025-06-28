import random
import numpy as np

#lưu ý: đây là phiên bản vô cùng đơn sơ của Neural Network
#phương pháp này sử dụng các phép nhân ma trận nên chạy trên CPU thông thường sẽ vô cùng chậm

class NeuralNetwork(object): #OOP related
    def __init__(self, sizes):
        #sizes là ma trận bao gồm số lượng neuron trong một layer 
        #ví dụ: sizes = [a, b, c] => layer 1 có a neuron, layer 2 có b neuron, layer 3 có c neuron
        #bias và weight được khởi tạo ngẫu nhiên trong khoảng từ 0 đến 1
        #layer đầu sẽ là input layer và bias ban đầu sẽ bằng 0 vì bias chỉ xuất hiện sau khi khởi tạo output của layer sau
        
        self.numLayers = len(sizes) #khai báo số lượng layer dựa theo sizes
        self.sizes = sizes #khai báo sizes của neuron
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #vì bias layer đầu là không tồn tại nên ta dùng: size[1:] (cho mọi layer trừ layer đầu tiên) 
        #y ở đây là độ dài cột của vector đầu vào, randn(y,1) = ma trận bias có kích thước y*1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        #ma trận weight có kích thước x*y , bỏ x ở lớp cuối và y ở lớp đầu

    def feedForward(self, x): #thêm hàm cho ra kết quả tiếp theo
        
        for b, w in zip(self.biases, self.weights): #b là hệ số bias, w là hệ số weight
            z = np.dot(w, x) + b
            x = self.sigmoid(z) #chạy qua 1 layer neuron
        
        return x #trả kết quả đầu ra
    
    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData = None):
        #train mô hình AI dựa trên phương pháp Stochastic Gradient Descent 
        #"trainingData" sẽ là những cụm (x, y) với x là dữ liệu input và y là output
        #"testData" là kết quả đầu ra tạm thời sau mỗi 1 vòng lặp => lúc đầu thì testData = None
        #"Epoch" sẽ là số lần trải qua 1 vòng lặp 
        #"miniBatchSize" sẽ là kích cỡ của mini batch trong mỗi 1 bước sgd (mini batch bao gồm những giá trị x y trong các gói nhỏ khác nhau)
        #"eta" sẽ là tốc độ học
        
        if testData:
            n_test = len(testData) #nếu có kết quả thì biến length của testData sẽ được lưu trữ trong n_test
            n = len(trainingData) #length của trainingData được lưu trữ trong n
        
        for j in range(epochs): #với mỗi 1 vòng lặp
            random.shuffle(trainingData) #tráo đổi giá trị của training data để có thể tránh việc model bị thiên vị
            miniBatches = [trainingData[k : k + miniBatchSize] for k in range(0, n , miniBatchSize)] #số lượng mini batches #k chạy từ 0 đến n với bước nhảy bằng mini batch size
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            if testData: 
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), n_test)) #Epoch lần thứ J: số lần thành công / số lần tổng
            else:
                print ("Epoch {0} complete".format(j)) #chạy vòng lặp cuối cùng

    def updateMiniBatch(self, miniBatch, eta):
        #hàm sửa đổi weight và bias bằng cách sử dụng gradient descent từ backpropagation với 1 mini batch duy nhất
        #miniBatch bao gồm input x và output y
        
        gradient_b = [np.zeros(b.shape) for b in self.biases] #ban đầu cho tổng gradient của tất cả bằng 0 với kích thước bằng kích thước của weight và bias ban đầu
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in miniBatch:
            delta_gradient_b, delta_gradient_w = self.backprop(x,y) #giá trị gradient đơn của từng bias và weight được lưu trữ trong self.backprop
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)] #tổng gradient cost function
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        
        self.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, gradient_w)] #cập nhật lại các biến số dựa vào công thức w := w - (eta / m) * tổng gradient
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, gradient_b)]

    def backprop(self, x, y):
        #hàm cho bước back propagation
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        #feed forward
        activation = x 
        activations = [x] #lưu trữ tất cả giá trị đầu ra, hay activations, theo từng lớp
        zs = [] #lưu trữ lại các vector z, theo từng lớp (z là giá trị đầu ra sau khi chạy qua phương trình feedforward sigmoid(x*w + b))
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b 
            zs.append(z) #chèn kết quả trên vào array
            activation = self.sigmoid(z) #chính là x
            activations.append(activation) #array x 
        #backward pass
        delta = self.costDer(activations[-1], y) * self.sigmoidDer(zs[-1]) #đây là công thức của delta: hàm tính độ chính xác của layer cuối cùng. ta lấy giá trị cuối của activation x và lấy giá trị cuối của z
        gradient_b[-1] = delta #gradient bias của layer cuối sẽ bằng cho delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose()) #gradient weight của layer cuối sẽ là dot product của delta và layer số trước số cuối ^ transpose vì ta muốn ma trận gradient weight có hình dạng của ma trận weight ban đầu
        #cho hàm chạy tiếp về hướng ngược lại sau khi đi qua layer cuối cùng
        
        for l in range(2, self.numLayers):
            z = zs[-l] #lấy giá trị z cuối cùng
            placeholderActivation = self.sigmoidDer(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * placeholderActivation
            gradient_b[-l] = delta
            gradient_w[-l] = np.dot(delta, activations[-l-1].transpose()) #-l-1 bởi vì l bắt đầu bằng 2 thì gradient b lấy layer -2, gradient w lấy layer -3 hay (-l, -l-1)
        return (gradient_b, gradient_w)
    
    def evaluate(self, testData):
        #hàm tìm số lần mà model trả về kết quả chính xác
        #lưu ý: kết quả của model được cho là kết quả trong layer neuron cuối cùng với giá trị activation x cao nhất
        results = [(np.argmax(self.feedForward(x)), y) for (x, y) in testData] #cho chạy giá trị x qua hàm feedforward lần cuối trước khi so sánh với y #argmax là giá trị lớn nhất của tất cả giá trị x
        return sum(int(x == y) for (x, y) in results) #nếu mà x == y thì sẽ +1 lần, sum sẽ là tổng số lần chính xác

    def costDer(self, outputActivation, y):
        #hàm trả lại gradient của cost function với outputActivation là kết quả máy trả về và y là kết quả thực sự
        return (outputActivation - y)

    # hàm tính toán

    def sigmoid(self, z):
        return 1/(1+np.exp(-z)) #Hàm phi tuyến tính và exp là của np nên ghi np.exp
    
    def sigmoidDer(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z)) #Đạo hàm của hàm phi tuyến
