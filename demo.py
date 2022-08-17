import cv2
import numpy as np
from math import cos, sin
import torch
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import time
# from cv2 import dnn_superres
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox
sys.path.append('..')
# from yolov5x.HEAD_ESTIMATION.lib.FSANET_model import *

# from keras.layers import Average

class Detector:
    def __init__(self,
        weights,  # model.pt path(x)
        classes = None,  # filter by class: --class 0, or --class 0 2 3
        imgsz=640,  # inference size (pixels)
         img_size=64,
         img_idx=0,
         skip_frame=1,  # every 5 frame do 1 detection and network forward propagation
         ad=0.6,
         # Parameters
         num_capsule=3,
         dim_capsule=16,
         routings=2,
         stage_num=[3, 3, 3],
         lambda_d=1,
         num_classes=3,
         image_size=64,
         num_primcaps=7 * 3,
         m_dim=5,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        dnn=False,  # use OpenCV DNN for ONNX inference
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        save_txt=False,
        hide_labels=False,
        hide_conf=False
        ):
        self.weights = weights
        self.imgsz = imgsz
        self.max_det = max_det
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.dnn = dnn
        self.image = image
        self.hide_labels = hide_labels
        self.hide_conf =  hide_conf
        self.img_size = img_size
        self.img_idx = img_idx
        self.skip_frame = skip_frame
        self.ad = ad
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.stage_num = stage_num
        self.lambda_d = lambda_d
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_primcaps = num_primcaps
        self.m_dim = m_dim
        self.name = name
        self.exist_ok = exist_ok
        self.save_txt = save_txt

    # Directories
        self.save_dir = increment_path(Path(project) / self.name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    def Prediction_1(self, im0s):
        source = str(im0s)
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s,vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp32
            im = im / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = self.model(im, augment=self.augment, visualize = self.visualize)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 2, self.agnostic_nms, max_det=self.max_det)
            self.pred_list = []
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, self.im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(self.im0, line_width=False, example=str(False))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], self.im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        crop_box_areas = [x1, y1, x2, y2]
                        self.pred_list.append([int(i) for i in np.array(crop_box_areas)])


            return self.pred_list
    def Mobile_detection(self):
        # for detection in self.pred_list_3:
        #     output = list(detection[:4])
        #     output = [int(x) for x in output]
        #     cropped_image = self.im0[output[1]:output[3], output[0]:output[2]]
        for detection in self.pred_list:
            output = list(detection[:4])
            output = [int(x) for x in output]
            xmax1 = int(((output[2] - output[0]) * 0.41) + output[0])
            cropped_image = self.im0[output[1]:output[3], xmax1:output[2]]
            img = letterbox(cropped_image, self.imgsz, stride=self.stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp32
            im = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = self.model(im, augment=self.augment, visualize=self.visualize)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 67, self.agnostic_nms,
                                       max_det=self.max_det)
            pred_list_4 = []
            for i, det in enumerate(pred):  # per image
                s, im0 = '', cropped_image.copy()
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=False, example=str(False))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        annotator.box_label(xyxy, label=False, color=colors(c, True))
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        crop_box_areas_2 = [x1, y1, x2, y2]
                        pred_list_4.append([int(i) for i in np.array(crop_box_areas_2)])
                        label = None if self.hide_labels else (
                            self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(crop_box_areas_2, label, color=colors(c, True))
                        # if c == 67:
                            # text = "PHONE USED"
                            # cv2.putText(self.im0, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for det in pred_list_4:
                output2 = list(det[:4])
                output2 = [int(x) for x in output2]
                # new_coors = [output[0] + output2[0],
                #              output[1] + output2[1],
                #              (output2[2] - output2[0]) + output[0] + output2[0],
                #              (output2[3] - output2[1]) + output[1] + output2[1]]
                new_coors = [xmax1 + output2[0] - 2,
                              output[1] + output2[1] - 2,
                              (output2[2] - output2[0]) + xmax1 + output2[0] + 1,
                              (output2[3] - output2[1]) + output[1] + output2[1] + 1]
                imagess = cv2.rectangle(self.im0, tuple(new_coors[:2]), tuple(new_coors[2:]), (0, 0, 255), 1)
                cv2.imwrite("AAAA.jpg", imagess)
    def mainA(self, video):
        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        size = (width, height)
        out = cv2.VideoWriter('driver_video.avi', codec, 20.0, size)

        while cap.isOpened():
            _, frame = cap.read()

            Detector.Prediction_1(self,frame)
            Detector.Prediction_2(self)
            # Detector.Mobile_detection(self)
            input_img = Detector(self)


            out.write(input_img)
            cap.release()
            cv2.destroyAllWindows()


video_path = "a.mp4"
weights_path = 'yolov5x6.pt'
image = 'mobile_phone.jpg'
y = Detector(weights_path)
y.Prediction_1(image)
y.Mobile_detection()
# y.mainA(video_path)


