import numpy as np
def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    b1 = np.array(np.tile((0, 1), n_cols // 2, ), dtype='bool')
    b2 = np.zeros(n_cols, dtype='bool')
    r2 = np.array(np.tile((1, 0), n_cols // 2), dtype='bool')
    r1 = np.zeros(n_cols, dtype='bool')
    g1 = np.array(np.tile((1, 0), n_cols // 2), dtype='bool')
    g2 = np.array(np.tile((0, 1), n_cols // 2), dtype='bool')
    if (n_cols % 2) == 1:
        b1 = np.append(b1, 0)
        r2 = np.append(r2, 1)
        g1 = np.append(g1, 1)
        g2 = np.append(g2, 0)
    b = b1
    r = r1
    g = g1
    for i in range(1, n_rows):
        if i % 2 == 0:
            b = np.array(np.dstack((b, b1)), dtype='bool')
            r = np.array(np.dstack((r, r1)), dtype='bool')
            g = np.array(np.dstack((g, g1)), dtype='bool')
        else:
            b = np.array(np.dstack((b, b2)), dtype='bool')
            r = np.array(np.dstack((r, r2)), dtype='bool')
            g = np.array(np.dstack((g, g2)), dtype='bool')
    r3 = r[0, :, :]
    g3 = g[0, :, :]
    b3 = b[0, :, :]
    ans = np.stack((r3, g3, b3), axis=2)
    return ans.astype(np.bool)


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    a, b = raw_img.shape
    x2 = get_bayer_masks(b,a)
    ans = raw_img[:, :, None] * x2
    return ans


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    a=np.zeros(colored_img.shape[0:2],dtype=np.uint8)
    c,d = colored_img.shape[0:2]
    for i in range(3):
        a=a+colored_img[:,:,i]*get_bayer_masks(d,c)[:,:,i]
    return a.astype(np.uint8)


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    img = raw_img
    ans=get_colored_img(img)
    r = ans[..., 0].astype(np.float32)
    g = ans[..., 1].astype(np.float32)
    bl = ans[..., 2].astype(np.float32)
    a, b, c = ans.shape
    ms = get_bayer_masks(b, a)
    rcof = ms[..., 0].astype(np.uint, copy=False)
    gcof = ms[..., 1].astype(np.uint, copy=False)
    bcof = ms[..., 2].astype(np.uint, copy=False)

    # Red amount in each pixel
    rcof[:a - 1, :] += ms[1:a, :, 0]
    rcof[1:a, :] += ms[0:a - 1, :, 0]
    rcof[:, 0:(b - 1)] += ms[:, 1:b, 0]
    rcof[:, 1:b] += ms[:, 0:b - 1, 0]
    rcof[:a - 1, :b - 1] += ms[1:, 1:, 0]
    rcof[1:, 1:] += ms[:a - 1, :b - 1, 0]
    rcof[:a - 1, 1:] += ms[1:, :b - 1, 0]
    rcof[1:, :b - 1] += ms[:a - 1, 1:, 0]

    # Blue amount in each pixel
    bcof[:a - 1, :] += ms[1:a, :, 2]
    bcof[1:, :] += ms[0:a - 1, :, 2]
    bcof[:, 0:b - 1] += ms[:, 1:b, 2]
    bcof[:, 1:b] += ms[:, 0:b - 1, 2]
    bcof[:a - 1, :b - 1] += ms[1:, 1:, 2]
    bcof[1:, 1:] += ms[:a - 1, :b - 1, 2]
    bcof[:a - 1, 1:] += ms[1:, :b - 1, 2]
    bcof[1:, :b - 1] += ms[:a - 1, 1:, 2]

    # Green amount in each pixel
    gcof[:a - 1, :] += ms[1:a, :, 1]
    gcof[1:, :] += ms[0:a - 1, :, 1]
    gcof[:, 0:b - 1] += ms[:, 1:b, 1]
    gcof[:, 1:b] += ms[:, 0:b - 1, 1]

    ans = (ans.astype(np.float32))

    # Red sum in each pixel
    ans[:a - 1, :, 0] += r[1:a, :]
    ans[1:, :, 0] += r[0:a - 1, :]
    ans[:, 0:b - 1, 0] += r[:, 1:b]
    ans[:, 1:b, 0] += r[:, 0:b - 1]
    ans[:a - 1, :b - 1, 0] += r[1:, 1:]
    ans[1:, 1:, 0] += r[:a - 1, :b - 1]
    ans[:a - 1, 1:, 0] += r[1:, :b - 1]
    ans[1:, :b - 1, 0] += r[:a - 1, 1:]

    # Blue sum in each pixel
    ans[:a - 1, :, 2] += bl[1:a, :]
    ans[1:, :, 2] += bl[0:a - 1, :]
    ans[:, 0:b - 1, 2] += bl[:, 1:b]
    ans[:, 1:b, 2] += bl[:, 0:b - 1]
    ans[:a - 1, :b - 1, 2] += bl[1:, 1:]
    ans[1:, 1:, 2] += bl[:a - 1, :b - 1]
    ans[:a - 1, 1:, 2] += bl[1:, :b - 1]
    ans[1:, :b - 1, 2] += bl[:a - 1, 1:]

    # Green sum in each pixel
    ans[:a - 1, :, 1] += g[1:a, :]
    ans[1:, :, 1] += g[0:a - 1, :]
    ans[:, 0:b - 1, 1] += g[:, 1:b]
    ans[:, 1:b, 1] += g[:, 0:b - 1]

    # Calculating mid
    ans[..., 0] /= rcof
    ans[..., 1] /= gcof
    ans[..., 2] /= bcof

    ans = (ans).clip(0, 255).astype(np.uint8)
    return ans


def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    ans = get_colored_img(raw_img)
    r = ans[..., 0].astype(np.float64)
    g = ans[..., 1].astype(np.float64)
    bl = ans[..., 2].astype(np.float64)
    mss = np.stack((r, g, bl), axis=2)
    a, b, c = ans.shape
    ms = get_bayer_masks(b,a)
    rcof = ms[..., 0].astype(np.float64, copy=False)
    gcof = ms[..., 1].astype(np.float64, copy=False)
    bcof = ms[..., 2].astype(np.float64, copy=False)
    ans = (ans.astype(np.float64))

    # Blue amount in each pixel
    # sh_lin
    bcof[:a - 1, :] += ms[1:a, :, 2] * 4
    bcof[1:, :] += ms[0:a - 1, :, 2] * 4
    bcof[:, :b - 1] += ms[:, 1:, 2] * 4
    bcof[:, 1:b] += ms[:, 0:b - 1, 2] * 4

    # sh_diag
    bcof[:a - 1, :b - 1] += ms[1:, 1:, 2] * 2
    bcof[1:, 1:] += ms[:a - 1, :b - 1, 2] * 2
    bcof[:a - 1, 1:] += ms[1:, :b - 1, 2] * 2
    bcof[1:, :b - 1] += ms[:a - 1, 1:, 2] * 2

    # Blr
    bcof += rcof * 6
    bcof[:a - 2, :] -= ms[2:, :, 0] / 2 * 3
    bcof[2:, :] -= ms[:a - 2, :, 0] / 2 * 3
    bcof[:, :b - 2] -= ms[:, 2:, 0] / 2 * 3
    bcof[:, 2:] -= ms[:, :b - 2, 0] / 2 * 3

    # Bl_gr_lin
    bcof[:a - 1, :b - 1] -= ms[1:, 1:, 1]
    bcof[1:, 1:] -= ms[:a - 1, :b - 1, 1]
    bcof[:a - 1, 1:] -= ms[1:, :b - 1, 1]
    bcof[1:, :b - 1] -= ms[:a - 1, 1:, 1]

    # Bl_gr_diag
    bcof += gcof * 5
    bcof[2::2, ::2] -= ms[:a - 2:2, ::2, 1]
    bcof[:a - 2:2, ::2] -= ms[2::2, ::2, 1]
    bcof[::2, 2::2] += ms[::2, :b - 2:2, 1] / 2
    bcof[::2, :b - 2:2] += ms[::2, 2::2, 1] / 2

    # Bl_gr_nechet
    bcof[3:a - 1:2, 1::2] += ms[1:a - 3:2, 1::2, 1] / 2
    bcof[1:a - 3:2, 1::2] += ms[3:a - 1:2, 1::2, 1] / 2
    bcof[1::2, 3:b - 1:2] -= ms[1::2, 1:b - 3:2, 1]
    bcof[1::2, 1:b - 3:2] -= ms[1::2, 3:b - 1:2, 1]

    # Red amount in each pixel
    # sh_lin
    rcof[:a - 1, :] += ms[1:a, :, 0] * 4
    rcof[1:, :] += ms[0:a - 1, :, 0] * 4
    rcof[:, :b - 1] += ms[:, 1:, 0] * 4
    rcof[:, 1:b] += ms[:, 0:b - 1, 0] * 4

    # sh_diag
    rcof[:a - 1, :b - 1] += ms[1:, 1:, 0] * 2
    rcof[1:, 1:] += ms[:a - 1, :b - 1, 0] * 2
    rcof[:a - 1, 1:] += ms[1:, :b - 1, 0] * 2
    rcof[1:, :b - 1] += ms[:a - 1, 1:, 0] * 2

    # Red_bl
    rcof += ms[..., 2] * 6
    rcof[:a - 2, :] -= ms[2:, :, 2] / 2 * 3
    rcof[2:, :] -= ms[:a - 2, :, 2] / 2 * 3
    rcof[:, :b - 2] -= ms[:, 2:, 2] / 2 * 3
    rcof[:, 2:] -= ms[:, :b - 2, 2] / 2 * 3

    # Red_gr_lin
    rcof += gcof * 5

    # R_gr_diag
    rcof[:a - 1, :b - 1] -= ms[1:, 1:, 1]
    rcof[1:, 1:] -= ms[:a - 1, :b - 1, 1]
    rcof[:a - 1, 1:] -= ms[1:, :b - 1, 1]
    rcof[1:, :b - 1] -= ms[:a - 1, 1:, 1]

    # R_chet
    rcof[2::2, ::2] += ms[:a - 2:2, ::2, 1] / 2
    rcof[:a - 2:2, ::2] += ms[2::2, ::2, 1] / 2
    rcof[::2, 2::2] -= ms[::2, :b - 2:2, 1]
    rcof[::2, :b - 2:2] -= ms[::2, 2::2, 1]

    # R-gr_nechet
    rcof[3:a - 1:2, 1::2] -= ms[1:a - 3:2, 1::2, 1]
    rcof[1:a - 3:2, 1::2] -= ms[3:a - 1:2, 1::2, 1]
    rcof[1::2, 3:b - 1:2] += ms[1::2, 1:b - 3:2, 1] / 2
    rcof[1::2, 1:b - 3:2] += ms[1::2, 3:b - 1:2, 1] / 2

    # Green amount in each pixel
    gcof[:a - 1, :] += ms[1:a, :, 1] * 2.
    gcof[1:, :] += ms[0:a - 1, :, 1] * 2
    gcof[:, 0:b - 1] += ms[:, 1:b, 1] * 2
    gcof[:, 1:b] += ms[:, 0:b - 1, 1] * 2
    gcof += ms[..., 0] * 4
    gcof[2:, :] -= ms[:a - 2, :, 0]
    gcof[:a - 2, :] -= ms[2:, :, 0]
    gcof[:, :b - 2] -= ms[:, 2:, 0]
    gcof[:, 2:] -= ms[:, :b - 2, 0]
    gcof += ms[..., 2] * 4
    gcof[2:, :] -= ms[:a - 2, :, 2]
    gcof[:a - 2, :] -= ms[2:, :, 2]
    gcof[:, :b - 2] -= ms[:, 2:, 2]
    gcof[:, 2:] -= ms[:, :b - 2, 2]

    # Red sum in each pixel
    # sh_lin
    ans[:a - 1, :, 0] += mss[1:a, :, 0] * 4
    ans[1:, :, 0] += mss[0:a - 1, :, 0] * 4
    ans[:, :b - 1, 0] += mss[:, 1:, 0] * 4
    ans[:, 1:b, 0] += mss[:, 0:b - 1, 0] * 4

    # sh_diag
    ans[:a - 1, :b - 1, 0] += mss[1:, 1:, 0] * 2
    ans[1:, 1:, 0] += mss[:a - 1, :b - 1, 0] * 2
    ans[:a - 1, 1:, 0] += mss[1:, :b - 1, 0] * 2
    ans[1:, :b - 1, 0] += mss[:a - 1, 1:, 0] * 2

    # Red_bl
    ans[..., 0] += bl * 6
    ans[:a - 2, :, 0] -= mss[2:, :, 2] * 3 / 2
    ans[2:, :, 0] -= mss[:a - 2, :, 2] * 3 / 2
    ans[:, :b - 2, 0] -= mss[:, 2:, 2] * 3 / 2
    ans[:, 2:, 0] -= mss[:, :b - 2, 2] * 3 / 2
    # Red_gr_lin
    ans[..., 0] += g * 5
    ans[:a - 1, :b - 1, 0] -= mss[1:, 1:, 1]
    ans[1:, 1:, 0] -= mss[:a - 1, :b - 1, 1]
    ans[:a - 1, 1:, 0] -= mss[1:, :b - 1, 1]
    ans[1:, :b - 1, 0] -= mss[:a - 1, 1:, 1]
    # Red_gr_diag
    ans[2::2, ::2, 0] += mss[:a - 2:2, ::2, 1] / 2
    ans[:a - 2:2, ::2, 0] += mss[2::2, ::2, 1] / 2
    ans[::2, 2::2, 0] -= mss[::2, :b - 2:2, 1]
    ans[::2, :b - 2:2, 0] -= mss[::2, 2::2, 1]
    # Red_gr_nechet
    ans[3:a - 1:2, 1::2, 0] -= mss[1:a - 3:2, 1::2, 1]
    ans[1:a - 3:2, 1::2, 0] -= mss[3:a - 1:2, 1::2, 1]
    ans[1::2, 3:b - 1:2, 0] += mss[1::2, 1:b - 3:2, 1] / 2
    ans[1::2, 1:b - 3:2, 0] += mss[1::2, 3:b - 1:2, 1] / 2

    # Blue sum in each pixel
    ans[:a - 1, :, 2] += mss[1:a, :, 2] * 4
    ans[1:, :, 2] += mss[0:a - 1, :, 2] * 4
    ans[:, :b - 1, 2] += mss[:, 1:, 2] * 4
    ans[:, 1:b, 2] += mss[:, 0:b - 1, 2] * 4

    # sh_diag
    ans[:a - 1, :b - 1, 2] += mss[1:, 1:, 2] * 2
    ans[1:, 1:, 2] += mss[:a - 1, :b - 1, 2] * 2
    ans[:a - 1, 1:, 2] += mss[1:, :b - 1, 2] * 2
    ans[1:, :b - 1, 2] += mss[:a - 1, 1:, 2] * 2

    # Blr
    ans[..., 2] += r * 6
    ans[:a - 2, :, 2] -= mss[2:, :, 0] / 2 * 3
    ans[2:, :, 2] -= mss[:a - 2, :, 0] / 2 * 3
    ans[:, :b - 2, 2] -= mss[:, 2:, 0] / 2 * 3
    ans[:, 2:, 2] -= mss[:, :b - 2, 0] / 2 * 3

    # Bl_gr_lin
    ans[:a - 1, :b - 1, 2] -= mss[1:, 1:, 1]
    ans[1:, 1:, 2] -= mss[:a - 1, :b - 1, 1]
    ans[:a - 1, 1:, 2] -= mss[1:, :b - 1, 1]
    ans[1:, :b - 1, 2] -= mss[:a - 1, 1:, 1]

    # Bl_gr_diag
    ans[..., 2] += g * 5
    ans[2::2, ::2, 2] -= mss[:a - 2:2, ::2, 1]
    ans[:a - 2:2, ::2, 2] -= mss[2::2, ::2, 1]
    ans[::2, 2::2, 2] += mss[::2, :b - 2:2, 1] / 2
    ans[::2, :b - 2:2, 2] += mss[::2, 2::2, 1] / 2

    # Bl_gr_nechet
    ans[3:a - 1:2, 1::2, 2] += mss[1:a - 3:2, 1::2, 1] / 2
    ans[1:a - 3:2, 1::2, 2] += mss[3:a - 1:2, 1::2, 1] / 2
    ans[1::2, 3:b - 1:2, 2] -= mss[1::2, 1:b - 3:2, 1]
    ans[1::2, 1:b - 3:2, 2] -= mss[1::2, 3:b - 1:2, 1]

    # Green sum in each pixel
    ans[:a - 1, :, 1] += g[1:, :] * 2
    ans[1:, :, 1] += g[:a - 1, :] * 2
    ans[:, :b - 1, 1] += g[:, 1:] * 2
    ans[:, 1:, 1] += g[:, :b - 1] * 2
    ans[..., 1] += r * 4
    ans[2:, :, 1] -= r[:a - 2, :]
    ans[:a - 2, :, 1] -= r[2:, :]
    ans[:, :b - 2, 1] -= r[:, 2:]
    ans[:, 2:, 1] -= r[:, :b - 2]
    ans[..., 1] += bl * 4
    ans[2:, :, 1] -= bl[:a - 2, :]
    ans[:a - 2, :, 1] -= bl[2:, :]
    ans[:, :b - 2, 1] -= bl[:, 2:]
    ans[:, 2:, 1] -= bl[:, :b - 2]

    # Calculating mid
    ans[..., 0] /= rcof
    ans[..., 1] /= gcof
    ans[..., 2] /= bcof
    ans = (ans).clip(0, 255).astype(np.uint8)
    return ans


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    img1 = img_pred
    img2 = img_gt
    a,b,c=img2.shape
    img1=img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse=(np.sum((img1-img2)*(img1-img2)))/a/b/c
    if mse==0:
        raise ValueError
    else:
        psnr=10*np.log10((np.max(img2*img2)/mse))
        return psnr


if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
