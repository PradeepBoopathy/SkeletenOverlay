import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

from model import HRNet
from util.joint_util import get_avg_preds
from faster_rcnn import predict as faster_rcnn_predict
from faster_rcnn import person_boxes


def get_affine(bbox, image_width, image_height):
    p1 = np.float32(
        [[int(bbox[0]), int(bbox[1])], [int(bbox[0] + bbox[2]), int(bbox[1])], [int(bbox[0]), int(bbox[1] + bbox[3])]])
    p2 = np.float32([[0, 0], [image_width, 0], [0, image_height]])
    M = cv2.getAffineTransform(p1, p2)
    return M


def preprocess_frame(frame, M, T, image_width, image_height):
    image_preprocess = cv2.warpAffine(frame, M, (image_width, image_height))
    image_transformed = T(image_preprocess)
    return image_transformed


def postprocess_predictions(preds, maxvals, M, clean_bbox_topleft, image_width, heatmap_width, image_height, heatmap_height):
    preds[:, 0] *= image_width / heatmap_width
    preds[:, 1] *= image_height / heatmap_height
    preds = np.matmul(np.linalg.inv(M[:, :2]), preds.T).T
    preds[:, 0] += clean_bbox_topleft[0]
    preds[:, 1] += clean_bbox_topleft[1]
    return preds, maxvals


def draw_skeleton(frame, preds, maxvals):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128)
    ]

    for i, (p, v) in enumerate(zip(preds, maxvals)):
        color = colors[i % len(colors)]
        if v[0] < 0.1:  # Check confidence score
            continue
        if is_within_frame(p[0], p[1], frame.shape[1], frame.shape[0]):
            cv2.circle(frame, (int(p[0]), int(p[1])), 1, color, 2)
        else:
            print(f"Point {i} with confidence {v[0]} is out of frame: ({p[0]}, {p[1]})")

    pred_int = preds.astype(int)

    joint_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16),
        (16, 18), (16, 20), (16, 22), (15, 17), (15, 19), (15, 21)
    ]

    for idx, (start, end) in enumerate(joint_pairs):
        if is_within_frame(preds[start, 0], preds[start, 1], frame.shape[1], frame.shape[0]) and \
           is_within_frame(preds[end, 0], preds[end, 1], frame.shape[1], frame.shape[0]):
            cv2.line(frame, tuple(pred_int[start]), tuple(pred_int[end]), colors[idx % len(colors)], 2)


def is_within_frame(x, y, frame_width, frame_height):
    return 0 <= x < frame_width and 0 <= y < frame_height


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    image_height = 384
    image_width = 288
    heatmap_height = 96
    heatmap_width = 72

    device = "cuda"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    T = transforms.Compose([transforms.ToTensor(), normalize])

    model = HRNet(base_channels=48, out_channels=23)
    model_dict = torch.load("weight/epoch_90/best_acc.pth")
    model.load_state_dict(model_dict['model_state_dict'])
    model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()
    model.eval()

    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800)
    faster_rcnn.eval().to(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.normalize(frame, frame, 0, 230, cv2.NORM_MINMAX)

        boxes, classes, labels = faster_rcnn_predict(frame, faster_rcnn, device, 0.8)
        frame, p_boxes = person_boxes(boxes, labels, frame)

        results = {'preds': [], 'maxvals': [], 'bbox': []}
        for p_box in p_boxes:
            w = p_box[2] - p_box[0]
            h = p_box[3] - p_box[1]
            scale = max([w / image_width, h / image_height]) * 1.4
            center = (p_box[0] + w / 2, p_box[1] + h / 2)
            clean_bbox_topleft = (center[0] - image_width / 2 * scale, center[1] - image_height / 2 * scale)
            clean_bbox = [clean_bbox_topleft[0], clean_bbox_topleft[1], image_width * scale, image_height * scale]

            M = get_affine(clean_bbox, image_width, image_height)
            image_transformed = preprocess_frame(frame, M, T, image_width, image_height)

            outputs = model(image_transformed.unsqueeze(0).to(device))

            preds, maxvals = get_avg_preds(outputs.detach().cpu().numpy(), 0.1)
            preds, maxvals = postprocess_predictions(preds[0], maxvals[0], M, clean_bbox_topleft, image_width, heatmap_width, image_height, heatmap_height)

            results['preds'].append(preds)
            results['maxvals'].append(maxvals)
            results['bbox'].append(clean_bbox)

        for pred, maxval, bbox in zip(results['preds'], results['maxvals'], results['bbox']):
            draw_skeleton(frame, pred, maxval)

        cv2.imshow("img", cv2.resize(frame, (1920, 1080)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
