import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# ФУНКЦІЯ ЗШИВАННЯ ДВОХ ФОТО
# ===============================

def stitch(img1, img2):

    # ORB ознаки
    orb = cv2.ORB_create(3000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        print("Недостатньо відповідностей")
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # RANSAC
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    result = cv2.warpPerspective(img2, H, (w1+w2, h1))
    result[0:h1,0:w1] = img1

    return result


# ===============================
# ЗАВАНТАЖЕННЯ ФОТО
# ===============================

img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")
#img3 = cv2.imread("img3.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)


# ===============================
# КРОК 1
# Зшиваємо img1 і img2
# ===============================

panorama12 = stitch(img1, img2)

# ===============================
# КРОК 2
# Додаємо img3
# ===============================

#panorama123 = stitch(panorama12, img3)


# ===============================
# ВІДОБРАЖЕННЯ
# ===============================

plt.figure(figsize=(15,7))
plt.imshow(panorama12)
plt.title("Panorama from 2 images")
plt.axis("off")
plt.show()