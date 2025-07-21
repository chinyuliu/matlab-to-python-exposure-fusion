from builtins import isinstance
import cv2
import numpy as np
from scipy.ndimage import convolve1d
import os
from typing import List

def exposure_fusion(image_stack: np.ndarray, params: list = [1.0, 1.0, 1.0]) -> np.ndarray:
    """
    對一系列曝光不同的影像進行融合。

    Args:
        image_stack (np.ndarray): 輸入的影像堆疊，維度為 (h, w, c, N)，其中 N 是影像數量。
                                影像應為 float64 型別，數值範圍 [0, 1]。
        params (list): 三個權重參數 [contrast, saturation, well-exposedness]。

    Returns:
        np.ndarray: 融合後的結果影像，數值範圍 [0, 1]。
    """
    h, w, _, num_images = image_stack.shape
    # 計算權重圖 (W)
    weights = np.ones((h, w, num_images), dtype=np.float64)
    contrast_p, sat_p, wexp_p = params
    
    if contrast_p > 0:
        weights *= _contrast(image_stack) ** contrast_p
    if sat_p > 0:
        weights *= _saturation(image_stack) ** sat_p
    if wexp_p > 0:
        weights *= _well_exposedness(image_stack) ** wexp_p
        
    weights += 1e-12  # 避免除以零
    
    # 正規化權重
    # np.sum(..., keepdims=True) 的功能類似 MATLAB 的 repmat
    weights /= np.sum(weights, axis=2, keepdims=True)
    
    # 影像金字塔融合
    # 建立一個空的拉普拉斯金字塔來儲存最終結果
    result_pyramid = laplacian_pyramid(np.zeros((h, w, 3), dtype=np.float64))
    n_levels = len(result_pyramid)
    print("n_levels:", n_levels)

    for i in range(num_images):
        img = image_stack[:, :, :, i]
        weight_map = weights[:, :, i]
        
        # 計算權重的高斯金字塔和影像的拉普拉斯金字塔
        weight_pyramid = gaussian_pyramid(weight_map, n_levels)
        img_pyramid = laplacian_pyramid(img, n_levels)
        
        # 進行融合
        for level in range(n_levels):
            # 權重圖是單通道，需擴展維度以匹配三通道影像
            w_l = weight_pyramid[level][..., np.newaxis] 
            result_pyramid[level] += w_l * img_pyramid[level]
            
    # 重建影像
    fused_image = reconstruct_laplacian_pyramid(result_pyramid)
    return np.clip(fused_image, 0, 1)

def _contrast(image_stack: np.ndarray) -> np.ndarray:
    """
    計算對比度權重 (最終調整版：手動灰階轉換以確保 float64 精度)
    """
    h, w, _, num_images = image_stack.shape
    output = np.zeros((h, w, num_images), dtype=np.float64)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)  # 濾波器
    
    for i in range(num_images):
        img_slice = image_stack[:, :, :, i].astype(np.float64)
        # 手動實現 rgb2gray
        # 標準加權平均公式: Y' = 0.299*R' + 0.587*G' + 0.114*B'
        mono = (img_slice[:, :, 0] * 0.299 + img_slice[:, :, 1] * 0.587 + img_slice[:, :, 2] * 0.114)
        
        # 使用 cv2.filter2D，並指定輸出深度為 CV_64F，確保濾波也在 float64 下完成
        contrast_map = cv2.filter2D(mono, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)
        output[:, :, i] = np.abs(contrast_map)
    '''
    for i in range(num_images):
        # OpenCV 的 BGR 順序與 MATLAB 的 rgb2gray 略有不同，但此處影響不大
        mono = cv2.cvtColor(image_stack[:, :, :, i].astype(np.float32), cv2.COLOR_RGB2GRAY)
        # 使用 cv2.filter2D 模擬 imfilter，邊界處理模式 'replicate' 對應 BORDER_REPLICATE
        output[:, :, i] = np.abs(cv2.filter2D(mono, -1, kernel, borderType=cv2.BORDER_REPLICATE))
    '''
    return output

def _saturation(image_stack: np.ndarray) -> np.ndarray:  # 飽和度
    h, w, _, num_images = image_stack.shape
    output = np.zeros((h, w, num_images), dtype=np.float64)
    for i in range(num_images):
        img = image_stack[:, :, :, i]
        # 沿著色彩通道計算標準差，即為飽和度
        output[:, :, i] = np.std(img, axis=2)
    return output

def _well_exposedness(image_stack: np.ndarray) -> np.ndarray:  # 曝光度
    # 將高斯函數應用於每個色彩通道
    gauss_curve = lambda x: np.exp(-0.5 * ((x - 0.5)**2) / (0.2**2))
    
    # 利用 NumPy 的廣播特性直接計算所有影像
    R = gauss_curve(image_stack[:, :, 0, :])
    G = gauss_curve(image_stack[:, :, 1, :])
    B = gauss_curve(image_stack[:, :, 2, :])
    return R * G * B

# --- 金字塔相關函式 ---

def _pyramid_filter() -> np.ndarray:
    """Burt and Adelson 的金字塔濾波器"""
    return np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])

def downsample(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """降採樣：先模糊再抽樣"""
    # Scipy 的 convolve1d 可以完美模擬 MATLAB 的 imfilter(I, filter, 'symmetric')
    # mode='mirror' 對應 MATLAB 的 'symmetric'
    blurry = convolve1d(img, kernel, axis=1, mode='mirror')
    blurry = convolve1d(blurry, kernel, axis=0, mode='mirror')
    # 每隔一個像素進行採樣
    return blurry[::2, ::2]

def upsample(img: np.ndarray, odd_dims: tuple, kernel: np.ndarray) -> np.ndarray:
    """
    升採樣：先插值再模糊 (修正版，對應 MATLAB 邏輯)。
    """
    # 擴增影像邊緣，對應 MATLAB 的 padarray(I,[1 1 0],'replicate')
    # NumPy 的 'edge' 模式等同於 MATLAB 的 'replicate'
    padded_img = np.pad(img, pad_width=((1, 1), (1, 1), (0, 0)) if img.ndim == 3 else ((1, 1), (1, 1)), mode='edge')

    # 建立一個基於 "擴增後" 尺寸的兩倍大影像
    h_padded, w_padded = padded_img.shape[:2]
    h_up, w_up = 2 * h_padded, 2 * w_padded
    
    upsampled = np.zeros((h_up, w_up) + padded_img.shape[2:], dtype=padded_img.dtype)
    
    # 將擴增後的影像像素填入偶數位置
    upsampled[::2, ::2] = 4 * padded_img  # 乘以 4 是為了補償濾波後的能量損失

    # 進行濾波模糊  由於邊緣已經擴增，這裡的 mode 影響不大，但仍保持 'mirror'
    blurry = convolve1d(upsampled, kernel, axis=1, mode='mirror')
    blurry = convolve1d(blurry, kernel, axis=0, mode='mirror')
    
    # 根據 MATLAB 的邏輯進行裁剪
    # MATLAB: R = R(3:r - 2 - odd(1), 3:c - 2 - odd(2), :)
    start_row, start_col = 2, 2
    end_row = h_up - 2 - odd_dims[0]
    end_col = w_up - 2 - odd_dims[1]
    return blurry[start_row:end_row, start_col:end_col]

def gaussian_pyramid(img: np.ndarray, n_levels: int = -1) -> List[np.ndarray]:
    """高斯金字塔"""
    if n_levels == -1:
        n_levels = int(np.floor(np.log2(min(img.shape[:2]))))
    
    pyramid = [img]
    kernel = _pyramid_filter()
    
    for _ in range(1, n_levels):
        img = downsample(img, kernel)
        pyramid.append(img)
    return pyramid

def laplacian_pyramid(img: np.ndarray, n_levels: int = -1) -> List[np.ndarray]:
    """拉普拉斯金字塔"""
    if n_levels == -1:
        n_levels = int(np.floor(np.log2(min(img.shape[:2]))))
        
    pyramid = []
    kernel = _pyramid_filter()
    current_img = img
    
    for _ in range(n_levels - 1):
        down = downsample(current_img, kernel)
        # 計算升採樣後需要裁剪的維度
        odd_dims = (2 * down.shape[0] - current_img.shape[0], 2 * down.shape[1] - current_img.shape[1])
        up = upsample(down, odd_dims, kernel)
        
        pyramid.append(current_img - up)
        current_img = down
    pyramid.append(current_img) # 最後一層是高斯金字塔的頂層
    return pyramid

def reconstruct_laplacian_pyramid(pyramid: List[np.ndarray]) -> np.ndarray:
    """從拉普拉斯金字塔重建影像"""
    n_levels = len(pyramid)
    kernel = _pyramid_filter()
    reconstructed_img = pyramid[n_levels - 1]
    
    for i in range(n_levels - 2, -1, -1):
        # 計算升採樣後需要裁剪的維度
        odd_dims = (2 * reconstructed_img.shape[0] - pyramid[i].shape[0], 
                    2 * reconstructed_img.shape[1] - pyramid[i].shape[1])
        up = upsample(reconstructed_img, odd_dims, kernel)
        reconstructed_img = pyramid[i] + up
    return reconstructed_img

def load_images(path: str, reduce: float = 1.0) -> np.ndarray:
    """從資料夾讀取影像序列，並進行對齊後輸出為 numpy 陣列"""
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])
    if not files:
        raise FileNotFoundError(f"在 '{path}' 中找不到任何影像檔案。")

    image_list = []
    for filename in files:
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：跳過無法讀取的影像 {filename}")
            continue

        # 轉為 RGB 並正規化到 [0, 1]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.astype(np.float64) / 255.0

        # 如果需要縮小尺寸
        if reduce < 1.0:
            h, w = int(img.shape[0] * reduce), int(img.shape[1] * reduce)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        image_list.append(img)

    if len(image_list) < 2:
        raise ValueError("需要至少兩張有效影像進行對齊。")

    # 呼叫 align_images 進行對齊
    aligned_list = align_images(image_list)

    # 將對齊後的圖片再轉回 RGB 並正規化為 float64，再堆疊成 numpy 陣列 (h, w, 3, N)
    aligned_stack = np.stack([
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        for img in aligned_list
    ], axis=-1)
    return aligned_stack

def align_images(images):
    # 確認輸入是否為至少兩張圖片的列表
    if not isinstance(images, list) or len(images) < 2:
        print("輸入必須是包含兩張以上圖片的列表")
        return None

    # 確認所有圖片尺寸是否一致
    size = images[0].shape
    for img in images:
        if img.shape != size:
            print("所有輸入圖片的尺寸必須相同")
            return None

    # 將圖片轉為灰階並正規化成 float32 格式，有助於提升對齊穩定性
    gray_images = [
        cv2.normalize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32),
            None, 0, 1, cv2.NORM_MINMAX
        )
        for img in images
    ]

    # 第一張圖作為對齊參考
    model_image = gray_images[0]
    sz = model_image.shape  # (height, width)

    # 建立對齊後的圖片列表，第一張圖片不需要對齊
    aligned_images = [images[0]]

    # 依序將第2張開始的圖片對齊到第一張圖片
    for i in range(1, len(images)):
        # 每張圖都重新初始化對齊參數矩陣（平移用）
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # 設定收斂條件（最大迭代次數與容許誤差）
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

        try:
            # 使用 ECC（Enhanced Correlation Coefficient）演算法計算對齊轉換矩陣
            (cc, warp_matrix) = cv2.findTransformECC(
                model_image,              # 參考圖片
                gray_images[i],           # 要對齊的圖片
                warp_matrix,              # 初始變換矩陣
                cv2.MOTION_TRANSLATION,   # 對齊模型：只考慮平移
                criteria,                 # 收斂條件
                None,                     # 不使用遮罩
                3                         # 高斯平滑的濾波器大小
            )

            # 使用剛剛計算出來的轉換矩陣對圖片進行仿射變換（warp）
            aligned = cv2.warpAffine(
                images[i],                # 原始彩色圖
                warp_matrix,              # 平移矩陣
                (sz[1], sz[0]),           # 輸出圖片大小 (width, height)
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

            # 加入對齊後的圖片
            aligned_images.append(aligned)

        except cv2.error as e:
            # 如果對齊失敗，顯示錯誤訊息，並保留原圖
            print(f"第 {i} 張圖片對齊失敗：{e}")
            aligned_images.append(images[i])
    return aligned_images

if __name__ == '__main__':
    # 建立一個名為 'house' 的資料夾，並放入你的影像
    image_folder = 'test_img' 
    
    if not os.path.isdir(image_folder):
        print(f"錯誤：請建立 '{image_folder}' 資料夾並將曝光序列影像放入其中")
    else:
        print("正在讀取影像...")
        # 讀取影像，此處不縮小 (reduce=1.0)
        I = load_images(image_folder, 1.0)

        print("正在進行曝光融合...")
        R = exposure_fusion(I, params=[1.0, 1.0, 1.0])  # contrast, saturation, well-exposedness
        print("融合完成")
        R_8bit = (R * 255).astype(np.uint8)
        
        # 顯示結果
        #cv2.imshow('Fused Result', cv2.cvtColor(R_8bit, cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        # 儲存結果
        cv2.imwrite('fusion_result.png', cv2.cvtColor(R_8bit, cv2.COLOR_RGB2BGR))
