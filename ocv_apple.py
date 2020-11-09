import tkinter as tk
import cv2
import numpy as np



def anime_filter(img, K=20):
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # ぼかしでノイズ低減
    edge = cv2.blur(gray, (3, 3))

    # Cannyアルゴリズムで輪郭抽出
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)

    # 輪郭画像をRGB色空間に変換
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # 画像の減色処理
    img = np.array(img/K, dtype=np.uint8)
    img = np.array(img*K, dtype=np.uint8)

    # 差分を返す
    return cv2.subtract(img, edge)

def anime_Button():
    # 入力画像の読み込み
    img = cv2.imread("s37P02yT_400x400.jpg")

    # 画像のアニメ絵化
    anime = anime_filter(img)

    # 結果出力
    cv2.imshow("input", anime)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__anime_Button__':
    anime_Button()

# 減色処理
def sub_color(src, K):
    # 次元数を1落とす
    Z = src.reshape((-1,3))

    # float32型に変換
    Z = np.float32(Z)

    # 基準の定義
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-means法で減色
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # UINT8に変換
    center = np.uint8(center)

    res = center[label.flatten()]

    # 配列の次元数と入力画像と同じに戻す
    return res.reshape((src.shape))


# モザイク処理
def mosaic(img, alpha):
    # 画像の高さ、幅、チャンネル数
    h, w, ch = img.shape

    # 縮小→拡大でモザイク加工
    img = cv2.resize(img,(int(w*alpha), int(h*alpha)))
    img = cv2.resize(img,(w, h), interpolation=cv2.INTER_NEAREST)

    return img


# ドット絵化
def pixel_art(img, alpha=2, K=4):
    # モザイク処理
    img = mosaic(img, alpha)

    # 減色処理
    return sub_color(img, K)

def pixel_Button():
    # 入力画像を取得
    img = cv2.imread("s37P02yT_400x400.jpg")

    # ドット絵化
    dst = pixel_art(img, 0.5, 4)
    
    # 結果を出力
    cv2.imshow("input", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__pixel_Button__':
    pixel_Button()

lbl = tk.Label(text="絵をたくさん加工しよう！")
anmbtn = tk.Button(text="アニメ化ボタン",command = anime_Button)
pixbtn = tk.Button(text="ドット化ボタン",command = pixel_Button)


root = tk.Tk()
root.geometry("400x400")

lbl.pack()
anmbtn.pack()
pixbtn.pack()

tk.mainloop
