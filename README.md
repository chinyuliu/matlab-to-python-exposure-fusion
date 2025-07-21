# 曝光融合（Exposure Fusion） - Python 實作版

本專案基於 2007 年 Pacific Graphics 會議論文  
**《Exposure Fusion》**（Tom Mertens, Jan Kautz, Frank Van Reeth）所提出之曝光融合技術，將 [Mericam/exposure-fusion](https://github.com/Mericam/exposure-fusion) 提供的 MATLAB 原始碼 **完整改寫為 Python 版本**。

本實作使用 NumPy、OpenCV 等基礎套件實現金字塔融合流程，並支援自動影像對齊（image alignment）、權重計算與最終融合，適合用於多張不同曝光度的照片合成高品質結果。

---

## 專案背景與實作說明

現代相機可拍攝多張不同曝光度的圖片，然而在單張圖片中往往難以同時保留亮區與暗部細節。  
**曝光融合（Exposure Fusion）** 是一種不需 HDR 輸出格式、僅透過影像融合即達到動態範圍擴展的技術。

本專案完整實作 Exposure Fusion 演算法，包括對齊前處理、特徵權重計算、Laplacian 金字塔建構與重建等步驟，忠實對應原始 MATLAB 程式行為。

---

## 實作流程與技術重點

1. **影像對齊**：自動計算平移參數，將輸入圖片對齊，避免錯位融合。
2. **權重計算**：根據三個特徵計算融合權重：
   - **對比度（Contrast）**
   - **飽和度（Saturation）**
   - **曝光適度（Well-exposedness）**
3. **權重正規化**：避免權重為零與 overflow，進行安全正規化。
4. **金字塔融合**：建構 Gaussian 與 Laplacian 金字塔，並進行多層融合。
5. **影像重建**：透過 Laplacian 金字塔重建出最終融合結果。

---

## 成果展示

| 輸入影像1 | 輸入影像2 | 輸入影像3 | 融合結果 |
|-----------|-----------|-----------|------------|
| ![](test_img/venice_under.png) | ![](test_img/venice_normal.png) | ![](test_img/venice_over.png) | ![](fusion_result.png) |


融合後的影像同時保留暗部與亮部細節，色彩自然。

<h2>成果展示</h2>

<!-- 輸入影像 -->
<h3>輸入影像</h3>
<table>
  <tr>
    <td><img src="test_img/peyrou_under.jpg" width="250"/></td>
    <td><img src="test_img/peyrou_mean.jpg" width="250"/></td>
    <td><img src="test_img/peyrou_over.jpg" width="250"/></td>
  </tr>
</table>

<!-- 融合結果 -->
<h3>融合結果</h3>
<table>
  <tr>
    <td><img src="fusion_result.png" width="760"/></td>
  </tr>
</table>

<p>融合後的影像同時保留暗部與亮部細節，色彩自然。</p>


---

## 快速開始

### 1. 安裝依賴

```bash
conda create -n exposure_fusion python=3.10.12 -y
conda activate exposure_fusion

pip install -r requirements.txt
```

### 2. 執行雨痕偵測程式

```bash
python3 exposure_fusion.py 
```

程式將輸出結果至指定資料夾。

---

## 目錄結構

```text
.
├── exposure_fusion.py    # 主程式
└── test_imgs/            # 放入需處理圖片

```

---

## 參考文獻與來源

- Mertens, T., Kautz, J., & Van Reeth, F. (2007).  
  **Exposure Fusion**. In Proceedings of the 15th Pacific Conference on Computer Graphics and Applications (PG'07) (pp. 382–390). IEEE.
  
- GitHub 開源程式碼參考：https://github.com/Mericam/exposure-fusion
---

