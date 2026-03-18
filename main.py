# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
python detectv1.py --weights yolov5s.pt --source 0 --classes 0
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow,
                           non_max_suppression, print_args, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

# ================= SETTINGS =================
ASPECT_RATIO = 1.2
HEIGHT_THRESHOLD = 150
FALL_FRAMES = 5   # number of consecutive frames to confirm fall

# ===========================================

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',
        source='0',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=True,
        classes=[0],  # person only
        half=False,
        dnn=False):

    source = str(source)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'http://', 'https://'))
    webcam = source.isnumeric() or (is_url and not is_file)

    if is_url and is_file:
        source = check_file(source)
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # ================= NEW VARIABLES =================
    fall_counter = 0
    prev_time = time_sync()
    # =================================================

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

    for path, im, im0s, vid_cap, s in dataset:

        # FPS CALCULATION FIX
        current_time = time_sync()
        loop_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / loop_time if loop_time > 0 else 0

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            if webcam:  # batch_size >= 1
                im0 = im0s[i].copy()
            else:
                im0 = im0s.copy()

            annotator = Annotator(im0, line_width=3, example=str(names))

            fall_detected = False

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    c = int(cls)  # integer class

                    # --- START FALL DETECTION LOGIC ---
                    if names[c] == 'person':
                        # Extract coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        w = x2 - x1
                        h = y2 - y1
                        r = w / h if h != 0 else 0

                        # CENTER POSITION
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # ===================== FALL LOGIC =====================
                        if r > ASPECT_RATIO and h < HEIGHT_THRESHOLD:
                            fall_counter += 1
                        else:
                            fall_counter = max(0, fall_counter - 1)

                        if fall_counter >= FALL_FRAMES:
                            fall_detected = True
                        # =======================================================

                        if fall_detected:
                            label = "FALL DETECTED!"
                            color = (0, 0, 255)

                            # ALERT TRIGGER
                            print("FALL ALERT!")
                            # TODO: Add code here to send an email or trigger alarm
                        else:
                            label = f"person {conf:.2f}"
                            color = colors(c, True)

                        annotator.box_label(xyxy, label, color=color)

            # Stream results
            im0 = annotator.result()

            if view_img:
                # FPS display
                cv2.putText(im0, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Fall warning text
                if fall_detected:
                    cv2.putText(im0, "!!! FALL DETECTED !!!", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                cv2.imshow("Fall Detection", im0)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--device', default='')
    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)