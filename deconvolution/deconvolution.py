import numpy as np
import matplotlib as plt
import scipy as sp

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    ans=np.zeros((size,size))
    
    if size%2==0:
        ans[0:size//2,0:size//2]+=1/(2*np.pi*sigma*sigma)
        for i in range(size//2):
            for j in range(size//2):
                r2=(size//2-1-i)**2+(size//2-j-1)**2
                ans[i,j]=ans[i,j]*(1/np.exp(r2/(2*sigma*sigma)))
        ans[size//2:size,0:size//2]=np.flip(ans[0:size//2,0:size//2],axis=0)
        ans[:,size//2:size]=np.flip(ans[:,0:size//2],axis=1)
    else:
        ans[0:size//2+1,0:size//2+1]+=1/(2*np.pi*sigma*sigma)
        for i in range(size//2+1):
            for j in range(size//2+1):
                r2=((size//2-i)**2)+((size//2-j)**2)
                ans[i,j]*=1/np.exp(r2/(2*sigma*sigma))
        ans[size//2+1:size,0:size//2+1]=np.flip(ans[0:size//2,0:size//2+1],axis=0)
        ans[size//2+1:size,size//2]=np.flip(ans[0:size//2,size//2],axis=0)
        ans[0:size,size//2+1:size]=np.flip(ans[:,0:size//2],axis=1)

    return ans/np.sum(ans)

def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    ans = np.zeros(shape,dtype=np.complex64)
    h=h.astype(np.complex64)
    a,b=h.shape
    n,m=shape
    ans[int(np.ceil((n-a)/2)):int(np.ceil((n-a)/2))+a,int(np.ceil((m-b)/2)):int(np.ceil((m-b)/2))+b]+=h
    ans=sp.fft.fft2(sp.fft.ifftshift(ans))
    return ans

def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H[np.abs(H)<=threshold]=0
    H[np.abs(H)>threshold]=1/H[np.abs(H)>threshold]
    return H

def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    blurred_img=blurred_img.astype(np.complex64)
    h=h.astype(np.complex64)
    g=sp.fft.fft2(blurred_img)
    ans=np.abs(sp.fft.ifft2(g*inverse_kernel(fourier_transform(h,blurred_img.shape),threshold)))
    return ans


def wiener_filtering(blurred_img, h, K=0.00009):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    g=sp.fft.fft2(blurred_img)
    H=fourier_transform(h,g.shape)
    H1=np.conj(H)
    f=np.abs(sp.fft.ifft2(g*H1/(H1*H+K)))
    return f


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    a,b=img1.shape
    ans=20*np.log10(255/np.sqrt((np.sum((img1-img2)*(img1-img2)/(a*b)))))
    return ans


