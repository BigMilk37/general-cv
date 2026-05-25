import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - (parameter_grad * self.lr)
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = updater.inertia * self.momentum + parameter_grad * self.lr
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        a = np.copy(inputs)
        a[a < 0] = 0
        return a
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        b = np.copy(self.forward_inputs)
        b[b >= 0] = 1
        b[b < 0] = 0
        return grad_outputs * b
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        a = np.zeros(inputs.shape)
        n = a.shape[0]
        for i in range(n):
            z = inputs[i]
            asss = np.exp(z - max(z))
            b = np.sum(np.exp((z - max(z))), axis=0)
            a[i] = asss / b
        return a
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        a = np.copy(self.forward_outputs)
        n, m = np.shape(grad_outputs)
        z = a * grad_outputs
        v = np.reshape(np.sum(a * grad_outputs, axis=1), (n, 1))
        u = a * v
        return z - u
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return np.matmul(self.weights,inputs.T).T + (self.biases)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        n, m = grad_outputs.shape
        self.weights_grad = (np.dot(np.transpose(grad_outputs), (self.forward_inputs)))
        if m == 1:
            self.biases_grad = np.array([np.sum(grad_outputs)])
        elif n == 1:
            self.biases_grad = grad_outputs[0]
        else:
            self.biases_grad = np.sum(grad_outputs, axis=0)
        return np.transpose(np.dot((self.weights).T,(grad_outputs).T))
        return np.transpose(np.dot((self.weights).T,(grad_outputs).T))
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        n, m = y_gt.shape
        return np.array([np.sum(-y_gt * np.log(y_pred)) / (n)])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        a = np.copy(y_pred)
        a[a <= eps] = eps
        n = y_gt.shape[0]
        return -y_gt / (a * n)
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    a = SGDMomentum(0.0001, 0.9)
    b = CategoricalCrossentropy()

    model = Model(b, a)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    c = (ReLU((784,np.array([784]))))
    c.input_shape = np.array([784])
    input_shape = np.array([784])
    model.add(c)
    # model.add(Dense(72))
    model.add(Dense((16)))
    model.add(Dense((16)))
    model.add(Dense(10))
    model.add(Softmax((10,1)))
    print(model)
    # 3) Train and validate the model using the provided data
    model.fit((x_train), (y_train), 6, 3, True, True, x_valid, y_valid)
    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    #your code here \/
    n,d,ih,iw=inputs.shape
    c,d,kh,kw=kernels.shape
    inputs_padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    bob=np.flip(kernels,(2,3))
    x=np.lib.stride_tricks.sliding_window_view(inputs_padded,(kh,kw),(2,3))
    ans = np.einsum('ndijkl,cdkl->ncij',x,bob)
    return ans
    # your code here


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
             (h, w) - image shape
        """
        # your code here \/
        a=((self.kernels.shape[3]))//2
        ans=convolve(inputs,self.kernels,a)+self.biases[None,:,None,None]

        return ans
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        a=(self.kernels_grad.shape[3]-1)//2
        self.biases_grad=np.sum(grad_outputs,axis=(0,2,3))
        b=np.moveaxis(self.forward_inputs,0,1)
        c=np.moveaxis(grad_outputs,0,1)
        self.kernels_grad=np.moveaxis(convolve(np.flip(b,(2,3)),(c),a),0,1)
        ans=convolve(grad_outputs,np.flip(np.moveaxis(self.kernels,0,1),(2,3)),a)
        return ans
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n,d,ih,iw=inputs.shape
        bob=inputs.reshape((n,d)+(ih//self.pool_size,)+(self.pool_size,iw//self.pool_size,self.pool_size))
        if self.pool_mode=='avg':
            a=np.average(bob,axis=(3,5))
        else:
            a = np.max(np.moveaxis(bob,3,4).reshape((n,d,ih//self.pool_size,iw//self.pool_size,self.pool_size**2)),axis=4)
            c=np.zeros((n,d,ih//self.pool_size,iw//self.pool_size,self.pool_size**2))
            d=np.argmax((np.moveaxis(bob,3,4).reshape((n,d,ih//self.pool_size,iw//self.pool_size,self.pool_size**2))),axis=4,keepdims=True)
            np.put_along_axis(c, d, 1, axis=4)
            self.forward_idxs = c
            # self.forward_idxs=(self.forward_idxs.reshape(self.forward_idxs.shape[:4]))!=0
        return a
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n,d,oh,ow=grad_outputs.shape
        y=oh*self.pool_size
        z=ow*self.pool_size
        x=self.pool_size
        bob = grad_outputs.repeat(x, axis=2).repeat(x, axis=3)
        if self.pool_mode=='max':
            a=(np.moveaxis(bob.reshape(n,d,oh,x,ow,x),3,4))*(self.forward_idxs.reshape(n,d,oh,ow,x,x))
            a=np.moveaxis(a,3,4).reshape(n,d,y,z)
        else:
            a=bob/x/x
        return a
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            m = np.mean(inputs, axis=(0, 2, 3))
            v = np.var(inputs, axis=(0, 2, 3))
            self.forward_centered_inputs=inputs-m[None,:,None,None]
            self.forward_inverse_std=1/((v[None,:,None,None]+eps)**0.5)
            self.forward_normalized_inputs=self.forward_centered_inputs*self.forward_inverse_std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * v
            ans=self.gamma[None,:,None,None]*self.forward_normalized_inputs+self.beta[None,:,None,None]
        else:
            self.forward_normalized_inputs=(inputs-self.running_mean[None,:,None,None])/((self.running_var[None,:,None,None]+eps)**0.5)
            ans=self.gamma[None,:,None,None]*self.forward_normalized_inputs+self.beta[None,:,None,None]
        return ans
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        m = grad_outputs.shape[0] * grad_outputs.shape[2] * grad_outputs.shape[3]
        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        dlxnorm = grad_outputs * self.gamma[None, :, None, None]
        dlsigma = np.sum(dlxnorm * self.forward_centered_inputs * (-1) / 2 * (self.forward_inverse_std ** 3),axis=(0, 2, 3))[None,:,None,None]
        dlu = np.sum(dlxnorm * (-1) * self.forward_inverse_std, axis=(0, 2, 3))[None,:,None,None] + dlsigma / m * np.sum((-2) * self.forward_centered_inputs, axis=(0, 2, 3))[None,:,None,None]
        dlx = dlxnorm * self.forward_inverse_std + dlsigma * 2 * self.forward_centered_inputs / m + dlu / m
        return dlx
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        a,b,c,d=inputs.shape
        return inputs.reshape(a,b*c*d)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        a,b=grad_outputs.shape
        return grad_outputs.reshape((a,)+self.input_shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask=(np.random.uniform(size=inputs.shape)>(self.p))
            a=self.forward_mask*inputs
        else:
            a=inputs*(1-self.p)
        return a
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs*self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    a = CategoricalCrossentropy()
    b = SGDMomentum(0.01, 0.9)
    model = Model(a,b)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)

    model.add(Conv2D(input_shape=(3,32,32),output_channels=32,kernel_size=3))
    model.add(BatchNorm(0.9))
    model.add(ReLU())
    model.add(Pooling2D(2, pool_mode='max'))

    model.add(Conv2D(output_channels=64,kernel_size=3))
    model.add(BatchNorm(0.9))
    model.add(Dropout(p=0.1))
    model.add(ReLU())
    model.add(Pooling2D(2, pool_mode='max'))

    model.add(Conv2D(output_channels=128,kernel_size=3))
    model.add(BatchNorm(0.9))
    model.add(ReLU())
    model.add(Pooling2D(2, pool_mode='max'))

    model.add(Flatten())
    model.add(Dropout(p=0.4))

    model.add(Dense(64))
    model.add(Dense(10))
    model.add(Softmax())
    print(model)
    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train,y_train=y_train,x_valid=x_valid,y_valid=y_valid,batch_size=16,epochs=3)
    # your code here /\
    return model


# ============================================================================
