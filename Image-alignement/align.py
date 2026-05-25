import numpy as np

def extract_channel_plates(raw_img, crop):
    raw_img=raw_img[0:raw_img.shape[0]-raw_img.shape[0]%3,:]
    b=raw_img[:raw_img.shape[0]//3,:]
    g=raw_img[raw_img.shape[0]//3:raw_img.shape[0]*2//3,:]
    r=raw_img[raw_img.shape[0]*2//3:raw_img.shape[0],:]
    unaligned_rgb=(r,g,b)
    ans=()
    if crop:
        for i in unaligned_rgb:
            h,w=i.shape
            ans+=(i[h//10:h-h//10,w//10:w-w//10],)
           
        coords=(np.array([h//10,w//10]),np.array([h+h//10,w//10]),np.array([2*h+h//10,w//10]))
        ans2=(coords[2],)+(coords[1],)+(coords[0],)
        return ans,ans2
    coords=(np.array([0,0]),np.array([raw_img.shape[0]//3,0]),np.array([raw_img.shape[0]*2//3,0]))
    ans2=(coords[2],)+(coords[1],)+(coords[0],)
    return unaligned_rgb, ans2

def find_relative_shift_pyramid(img_a, img_b):
    #Normalized cross correlation is slower than mse(13.9s vs 8.37) so I pick mse
    a = [img_a]
    b = [img_b]
    c = 1
    while np.max(a[0].shape)>500:
        c+=1
        img_a = np.array(img_a)[::2,::2]
        img_b = np.array(img_b)[::2,::2]
        a.insert(0,img_a)
        b.insert(0,img_b)
    a_to_b = np.array([0, 0])
    x = 15
    y = 15
    x0 = -15
    y0 = -15
    flag = 0
    for i in range(c):
        top_shift = np.array([0,0])
        top_mse = float('inf')
        min_norm = -1
        if flag:
            x0 = x
            y0 = y
        for j in range(x0,x+2):
            for k in range(y0,y+2):
                bob = np.roll(a[i],(j,k),axis = (0,1) )
                mse_score = np.sum((bob - b[i])**2)/bob.shape[0]/bob.shape[1]
                # norm_score = np.sum(bob*b[i])/np.sqrt(np.sum(bob**2)*np.sum(b[i]**2))
                if mse_score<top_mse:
                    top_mse = mse_score
                    top_shift = np.array([j,k])
                # if norm_score>min_norm:
                #     min_norm = norm_score
                #     top_shift = np.array([j,k])
        flag = 1
        a_to_b = top_shift
        x = 2*a_to_b[0]
        y = 2*a_to_b[1] 
    return a_to_b

def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):

    r = crops[0]
    g = crops[1]
    b = crops[2]
    r_to_g = find_relative_shift_fn(r,g)
    b_to_g = find_relative_shift_fn(b,g)
    r_to_g +=crop_coords[1] - crop_coords[0]
    b_to_g += crop_coords[1] - crop_coords[2]
    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    r, g, b = channels
    rcoord = channel_coords[0] + r_to_g  
    gcoord = channel_coords[1] 
    bcoord = channel_coords[2] + b_to_g            

    x_start = max(rcoord[0], gcoord[0], bcoord[0])
    x_end = min(rcoord[0] + r.shape[0], gcoord[0] + g.shape[0], bcoord[0] + b.shape[0])
    y_start = max(rcoord[1], gcoord[1], bcoord[1])
    y_end = min(rcoord[1] + r.shape[1], gcoord[1] + g.shape[1], bcoord[1] + b.shape[1])
    
    if x_start >= x_end or y_start >= y_end:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    r_crop = r[x_start - rcoord[0]: x_end - rcoord[0], y_start - rcoord[1]: y_end - rcoord[1]]
    g_crop = g[x_start - gcoord[0]: x_end - gcoord[0], y_start - gcoord[1]: y_end - gcoord[1]]
    b_crop = b[x_start - bcoord[0]: x_end - bcoord[0], y_start - bcoord[1]: y_end - bcoord[1]]

    if r_crop.shape != g_crop.shape or g_crop.shape != b_crop.shape or r_crop.shape != b_crop.shape:
    return np.dstack((r_crop, g_crop, b_crop))


def find_relative_shift_fourier(img_a, img_b):
    # Your code here
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    h, w = img_a.shape
    fft_a = np.fft.fft2(img_a)
    fft_a = np.conjugate(fft_a)
    fft_b = np.fft.fft2(img_b)
    cc = np.fft.ifft2(fft_a * fft_b)
    x_idx, y_idx = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    
    if x_idx > h//2:
        x_idx -= h
    if y_idx > w//2:
        y_idx -= w
    return np.array([x_idx, y_idx])


if __name__ == "__main__":

    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
