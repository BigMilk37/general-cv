from typing import List, Tuple

import numpy as np
import torch


class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float64,
    r_max: np.float64,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:

    s=((r_max-r_min)/(q_max-q_min)).astype(np.float64)
    z=np.round((r_max*q_min-r_min*q_max)/(r_max-r_min)).astype(np.int32)
    ans=QuantizationParameters(s,z,q_min,q_max)
    return ans




def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    return np.clip(np.round(r/qp.scale+qp.zero_point).astype(np.int32),a_min=qp.q_min,a_max=qp.q_max).astype(np.int8)



def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:

    return (qp.scale*(q.astype(np.int32)-qp.zero_point)).astype(np.float32)




class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float64).max
        self.max = np.finfo(np.float64).min

    def __call__(self, x: torch.Tensor):

        self.min = min(self.min,np.min(x.numpy()).astype(np.float64))
        self.max = max(self.max,np.max(x.numpy()).astype(np.float64))


def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    oaa=MinMaxObserver()
    oaa(torch.from_numpy(weights.astype(np.float64)))
    mini=oaa.min
    maxim=oaa.max
    if abs(mini)>maxim:
        maxim=abs(mini)
    else:
        mini=-maxim
    aooo=compute_quantization_params(mini,maxim,-127,127)
    quack=quantize(weights.astype(np.float64),aooo)
    return quack, aooo


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    r=weights[0].astype(np.float64)
    g=weights[1].astype(np.float64)
    # b=weights[2,:,:].astype(np.float64)
    o1=MinMaxObserver()
    o1(torch.from_numpy(r.astype(np.float64)))
    min1=o1.min
    max1=o1.max
    if abs(min1)>max1:
        max1=abs(min1)
    else:
        min1=-max1
    w1=compute_quantization_params(min1,max1,-127,127)
    o2=MinMaxObserver()
    o2(torch.from_numpy(g.astype(np.float64)))
    min2=o2.min
    max2=o2.max
    if abs(min2)>max2:
        max2=abs(min2)
    else:
        min2=-max2
    w2=compute_quantization_params(min2,max2,-127,127)
    # o3=MinMaxObserver()
    # o3(torch.from_numpy(b.astype(np.float64)))
    # min3=o3.min
    # max3=o3.max
    # if abs(min3)>max3:
    #     max3=abs(min3)
    # else:
    #     min3=-max3
    # w3=compute_quantization_params(min3,max3,-127,127)
    q1=quantize(r,w1)
    q2=quantize(g,w2)
    # q3=quantize(b,w3)
    ans1=np.vstack((q1[:,None,:,:],q2[:,None,:,:]),).astype(np.int8)
    return ans1,[w1,w2]


def quantize_bias(
    bias: np.float64,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    return ((np.round(bias/scale_w/scale_x).astype(np.int32)))


def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    a=str(m)
    b,c=a.split('.')
    if m>=0.5:
        n=0
        while (m>=0.5 and m>=1):
            m/=2
            n-=1
    else:
        n=0
        while (m<0.5):
            m*=2
            n+=1
    goodm=np.int32(np.round(2**31*m))
    n=np.int32(n)
    return n,goodm


def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    m=np.multiply(m0,float(2)**(-31),dtype=np.float64)
    m=np.multiply(m,float(2)**(-n),dtype=np.float64)
    real=np.multiply(accum,m,dtype=np.float64)
    res=np.multiply(accum,m0,dtype=np.int64)
    res=res//np.int32(2**n)
    bitp=('{0:064b}'.format(res))[36]
    fixed=33-n
    res=res//(2**(64-fixed-n))
    res=np.int32(res)
    if real>0:
        res=abs(res)
    else:
        res=-abs(res)
    if np.abs(np.float64(res)-real)>0.6:
        if real>=0:
            if np.float64(res)<real:
                res+=np.int32(1)
            else:
                res-=np.int32(1)
        if real<0:
            if np.float64(res)<real:
                res+=np.int32(1)
            else:
                res-=np.int32(1)
    return np.int32(res)
