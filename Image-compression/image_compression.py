import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    M=np.mean(matrix, axis=1)[:,None]
    m=matrix-M
    cov=np.cov(m)
    ev,evec=np.linalg.eigh(cov)
    order=np.argsort(-ev)
    sorted_vec = evec[:, order]
    evcut = sorted_vec[:, :p]
    ans = np.dot(evcut.T, m)

    return (evcut, ans, M[:,0])

def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    m1 = np.matmul(compressed[0][0], compressed[0][1])+compressed[0][2][:,None]
    m2 = np.matmul(compressed[1][0], compressed[1][1])+compressed[1][2][:,None]
    m3 = np.matmul(compressed[2][0], compressed[2][1])+compressed[2][2][:,None]
    return np.dstack([m1.clip(0, 255),m2.clip(0, 255),m3.clip(0, 255)])


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[:,:,j],p))
        decompressed = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    r=0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    g=-0.1687*img[:,:,0] + (-0.3313)*img[:,:,1] + 0.5*img[:,:,2]+128
    b=0.5*img[:,:,1] + (-0.4187*img[:,:,0])+  (-0.081*img[:,:,0])+128
    c=[]
    c.append(r)
    c.append(g)
    c.append(b)
    return np.dstack([r,g,b])


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    img=img.astype(np.float64)
    r=img[:,:,0]+1.402*(img[:,:,2]-128)
    g=img[:,:,0]+(-0.34414*(img[:,:,1]-128))+(-0.71414*(img[:,:,2]-128))
    b=img[:,:,0]+1.77*(img[:,:,1]-128)
    return np.dstack([r.clip(0,255).astype(np.uint8),g.clip(0,255).astype(np.uint8),b.clip(0,255).astype(np.uint8)])


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycc_img = rgb2ycbcr(rgb_img)
    ycc_img[:, :, 1:3] = gaussian_filter(ycc_img[:, :, 1:3], 10)
    rgb_img = ycbcr2rgb(ycc_img)
    plt.imshow(rgb_img)
    plt.savefig("gauss_1.png")

def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycc_img = rgb2ycbcr(rgb_img)
    ycc_img[:, :, 0] = gaussian_filter(ycc_img[:, :, 0], 10)
    rgb_img = ycbcr2rgb(ycc_img)
    plt.imshow(rgb_img)
    plt.savefig("gauss_2.png")

def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """
    a=gaussian_filter(component,10)

    return a[::2,::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

# Your code here
    # a=np.arange(0,8)
    ans=np.zeros((8,8))
    # # b=np.array([block[0,0]*np.cos(1/16/(2**0.5)*np.pi),np.sum(block[1,:]*np.cos((2*a[1:]+1)/16*np.pi)[:,None],axis=0)])
    # # c=np.sum(b*np.cos((2*a[1:]+1)/16*np.pi))+np.cos(b*np.cos(1/16/(2**0.5)*np.pi))
    # b=np.insert(block[1:, 1:] * np.cos((2 * a[1:] + 1) / 16 * np.pi)[:, None],0,block[0,1:]*np.cos(1/16*np.pi))
    # c=np.insert(np.cos((2*a[1:]+1)/16*np.pi)[None,:],0,np.cos(1/16*np.pi))
    # ans[1:,1:]=np.sum(np.sum(b,axis=0)*c)/4
    # d=np.copy(b)
    # d[0]=block[0,0]*np.cos(1/16*np.pi/(2**0.5))
    # ans[:,1:]=np.sum(np.sum(d,axis=0)*c)/4/(2**0.5)
    # c[0]=np.cos(1/16*np.pi/(2**0.5))
    # ans[1:]=np.sum(np.sum(b,axis=0)*c)/4/(2**0.5)
    # ans[0,0]=np.sum(np.sum(d,axis=0)*c)/8
    for i in range(8):
        for j in range(8):
            if i == 0:
                a = 1/np.sqrt(2)
            else: a = 1
            if j == 0:
                b = 1/np.sqrt(2)
            else: b = 1
            for k in range(8):
                for l in range(8):    
                    ans[i,j] += 0.25*a*b*block[k,l]*(np.cos((2*k+1)*i*(np.pi)/16))*(np.cos((2*l+1)*j*(np.pi)/16))
    return ans


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here
    ans = block / quantization_matrix
    ans = np.round(ans)
    return ans


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here
    if q<50: s = 5000/q
    elif q==100: s=1
    else: s = 200 - 2*q
    my_matrix = np.floor((default_quantization_matrix*s + 50)/100)
    my_matrix[my_matrix==0] = 1
    return my_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    ans = np.arange(64)
    c = 0
    flag = 1
    for i in range(15):
        if flag:  
            for str in range(7,-1,-1):
                if str > i: continue
                col = i-str
                if col>7: break
                ans[c] = block[str,col]
                c += 1
            flag = 0
        else: 
            for col in range(7,-1,-1):
                if col > i: continue
                str = i-col
                if str>7: break
                ans[c] = block[str,col]
                c += 1
            flag = 1

    return ans


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    a = []
    flag = 0
    c = 0
    for i in zigzag_list:
        if i != 0:
            if flag:
                a.append(0)
                a.append(c)
                c = 0
                flag = 0
            a.append(i)
        else:
            c+=1
            flag = 1
    if flag:
        a.append(0)
        a.append(c)
    return np.array(a).astype(np.float64)


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """


    # Переходим из RGB в YCbCr
    img = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    a = []
    a.append(img[...,0])
    a.append(downsampling(img[...,1]))
    a.append(downsampling(img[...,2]))
    img=a
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    y = img[0]
    c1 = img[1]
    c2 = img[2]
    y = (np.clip(y,0,256) - 128).astype(np.int8)
    a = []
    h,w =np.array(y).shape
    for i in range(0,h,8):
        for j in range(0,w,8):
            temp = compression(zigzag(quantization(dct(y[i:i+8,j:j+8]),y_quantization_matrix)))
            a.append(temp)
    c1 = (np.clip(c1,0,256) - 128).astype(np.int8)
    b = []
    h,w =np.array(c1).shape
    for i in range(0,h,8):
        for j in range(0,w,8):
            temp = compression(zigzag(quantization(dct(y[i:i+8,j:j+8]),color_quantization_matrix)))
            b.append(temp)
    c2 = (np.clip(c2,0,256) - 128).astype(np.int8)
    c = []
    h,w =np.array(c2).shape
    for i in range(0,h,8):
        for j in range(0,w,8):
            temp = compression(zigzag(quantization(dct(y[i:i+8,j:j+8]),color_quantization_matrix)))
            c.append(temp)
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    ans = []
    ans.append(a)
    ans.append(b)
    ans.append(c)

    return ans


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    a = []
    flag = 0
    for i in compressed_list:
        if i!=0:
            if flag:
                for j in range(int(i)):
                    a.append(0)
            else:
                a.append(i)
            flag = 0
        else:
            flag = 1
    return a


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    ans = np.zeros((8,8))
    c = 0
    flag = 1
    for i in range(15):
        if flag:  
            for str in range(7,-1,-1):
                if str > i: continue
                col = i-str
                if col>7: break
                ans[str,col] = input[c]
                c += 1
            flag = 0
        else: 
            for col in range(7,-1,-1):
                if col > i: continue
                str = i-col
                if str>7: break
                ans[str,col] = input[c]
                c += 1
            flag = 1

    return ans


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block*quantization_matrix

def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    ans=np.zeros((8,8))
    # # b=np.array([block[0,0]*np.cos(1/16/(2**0.5)*np.pi),np.sum(block[1,:]*np.cos((2*a[1:]+1)/16*np.pi)[:,None],axis=0)])
    # # c=np.sum(b*np.cos((2*a[1:]+1)/16*np.pi))+np.cos(b*np.cos(1/16/(2**0.5)*np.pi))
    # b=np.insert(block[1:, 1:] * np.cos((2 * a[1:] + 1) / 16 * np.pi)[:, None],0,block[0,1:]*np.cos(1/16*np.pi))
    # c=np.insert(np.cos((2*a[1:]+1)/16*np.pi)[None,:],0,np.cos(1/16*np.pi))
    # ans[1:,1:]=np.sum(np.sum(b,axis=0)*c)/4
    # d=np.copy(b)
    # d[0]=block[0,0]*np.cos(1/16*np.pi/(2**0.5))
    # ans[:,1:]=np.sum(np.sum(d,axis=0)*c)/4/(2**0.5)
    # c[0]=np.cos(1/16*np.pi/(2**0.5))
    # ans[1:]=np.sum(np.sum(b,axis=0)*c)/4/(2**0.5)
    # ans[0,0]=np.sum(np.sum(d,axis=0)*c)/8
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):    
                    if k == 0:
                        a = 1/np.sqrt(2)
                    else: a = 1
                    if l == 0:
                        b = 1/np.sqrt(2)
                    else: b = 1
                    ans[i,j] += 0.25*a*b*block[k,l]*(np.cos((2*i+1)*k*(np.pi)/16))*(np.cos((2*j+1)*l*(np.pi)/16))
    return np.round(ans)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """


    return np.repeat(np.repeat(component, 2, axis=0), 2, axis=1)


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    h,w,z = result_shape
    y = np.zeros((h,w))
    c1 = np.zeros((h//2,w//2))
    c2 = np.zeros((h//2,w//2))
    for i in range(h//8):
        for j in range(w//8):
            y[i*8:i*8+8,j*8:j*8+8] = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(result[0][i+j])),quantization_matrixes[0]))
    y+= 128
    for i in range(h//16):
        for j in range(w//16):
            c1[i*8:i*8+8,j*8:j*8+8] = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(result[1][i+j])),quantization_matrixes[1]))
            c2[i*8:i*8+8,j*8:j*8+8] = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(result[2][i+j])),quantization_matrixes[1]))
    c1+=128
    c2+=128
    c1=upsampling(c1)
    c2=upsampling(c2)
    a = []
    a.append(y)
    a.append(c1)
    a.append(c2)
    a = np.array(a)
    a = ycbcr2rgb(a)
    
    return a.astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        matr = [own_quantization_matrix(y_quantization_matrix, p), own_quantization_matrix(color_quantization_matrix, p)]
        compressed = jpeg_compression(img, matr)
        decomp = jpeg_decompression(compressed, img.shape, matr)
        axes[i // 3, i % 3].imshow(decomp)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")

def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig

def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


# if "main" == "main":
#     pca_visualize()
#     get_gauss_1()
#     get_gauss_2()
#     jpeg_visualize()
#     get_pca_metrics_graph()
#     get_jpeg_metrics_graph()