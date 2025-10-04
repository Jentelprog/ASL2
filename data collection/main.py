from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import csv

detector = HandDetector(
    staticMode=True, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5
)
head = [
    "image",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "x3",
    "y3",
    "z3",
    "x4",
    "y4",
    "z4",
    "x5",
    "y5",
    "z5",
    "x6",
    "y6",
    "z6",
    "x7",
    "y7",
    "z7",
    "x8",
    "y8",
    "z8",
    "x9",
    "y9",
    "z9",
    "x10",
    "y10",
    "z10",
    "x11",
    "y11",
    "z11",
    "x12",
    "y12",
    "z12",
    "x13",
    "y13",
    "z13",
    "x14",
    "y14",
    "z14",
    "x15",
    "y15",
    "z15",
    "x16",
    "y16",
    "z16",
    "x17",
    "y17",
    "z17",
    "x18",
    "y18",
    "z18",
    "x19",
    "y19",
    "z19",
    "x20",
    "y20",
    "z20",
    "x21",
    "y21",
    "z21",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "alpha",
]
# head = [
#     "image",
#     "p1",
#     "p2",
#     "p3",
#     "p4",
#     "p5",
#     "p6",
#     "p7",
#     "p8",
#     "p9",
#     "p10",
#     "p11",
#     "p12",
#     "p13",
#     "p14",
#     "p15",
#     "p16",
#     "p17",
#     "p18",
#     "p19",
#     "p20",
#     "p21",
#     "fu",
#     "alpha",
# ]
data = []
file_name = "data.csv"
for file in os.listdir("images"):
    result = file
    images_path = os.path.join("images", file)
    print(images_path)
    for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        img = cv2.imread(image_path)
        hand, img = detector.findHands(img, draw=True, flipType=True)
        l = []
        l.append(image)
        if hand:
            hand1 = hand[0]
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand1
            fingers1 = detector.fingersUp(hand1)
            for i in lmList1:
                for j in i:
                    l.append(j)
            for x in fingers1:
                l.append(x)
            l.append(result)
            print(l)
            data.append(l)

with open(file_name, "w", newline='') as file:
    csvWriter = csv.writer(file)
    csvWriter.writerow(head)
    csvWriter.writerows(data)

# img = cv2.imread("image.jpg")

# detector = HandDetector(
#     staticMode=True, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5
# )


# hand,img=detector.findHands(img,draw=True,flipType=True)


# cv2.imshow("wow",img)

# cv2.waitKey(5000)
