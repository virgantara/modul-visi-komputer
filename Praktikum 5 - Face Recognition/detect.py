from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20, device=device)  # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(
    device)  # initializing resnet for face img to embeding conversion

model = torch.load('data.pt')


def draw_bbox(bounding_boxes, image, label):
    for i in range(len(bounding_boxes)):
        x1, y1, x2, y2 = bounding_boxes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 2)

        pos = (int(x1+10), int(y1))
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image


def plot_landmarks(landmarks, image):
    for i in range(len(landmarks)):
        for p in range(landmarks[i].shape[0]):
            cv2.circle(image,
                       (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])),
                       2, (0, 0, 255), -1, cv2.LINE_AA)
    return image


def face_match(img):  # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.open(img_path)
    img = Image.fromarray(img)
    face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability

    name_list = model[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    if face is not None:
        if torch.cuda.is_available():
            face = face.cuda()

        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

        # saved_data = torch.load('data.pt')  # loading data.pt file
        embedding_list = model[0]  # getting embedding data

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))

        return (name_list[idx_min], min(dist_list))
    else:
        return None


# result = face_match('test_faqih.jpg')

cap = cv2.VideoCapture(0)

while 1:

    _, frame = cap.read()
    result = face_match(frame)
    if result is not None:
        # cv.imshow('res', result)
        bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True)
        if bounding_boxes is not None:
            # if (result[1]) < 0.8:
            frame = draw_bbox(bounding_boxes, frame, result[0])

            print(bounding_boxes)

            print('Face matched with: ', result[0], 'With distance: ', result[1])

    cv2.imshow("face", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

# print('Face matched with: ', result[0], 'With distance: ', result[1])
