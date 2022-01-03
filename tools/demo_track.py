import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


import sys
DARKNET_PATH = "/absolute/path/to/darknet"
sys.path.append(DARKNET_PATH)
assert os.path.exists(DARKNET_PATH), "Please set the correct path to darknet directory"

import darknet
import numpy as np
import datetime

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--output_dir", default=".\\", type=str, help="output dir")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=1, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    parser.add_argument("--weights", default="yolov4.weights", help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data", help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25, help="remove detections with confidence below this value")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Darknet_Predictor(object):
    def __init__(
        self,
        cfg,
        data,
        weights,
        thresh=0.05,
        batch_size=1,
    ):
        self.network, self.class_names, self.class_colors = darknet.load_network(
            cfg,
            data,
            weights,
            batch_size=batch_size
        )
        self.threshold = thresh
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.batch_size = batch_size

    def predict(self, frame, timer):
        raw_img = frame.copy()
        frame_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.width, self.height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        
        timer.tic()
        darknet_outputs = darknet.detect_image(self.network, self.class_names, img_for_detect, self.threshold)
        darknet.free_image(img_for_detect)
        
        if len(darknet_outputs) > 0:
            # outputs must be a N_detections x 5 array/tensor (xleft, ytop, xright, ybottom, confidence)
            outputs = [
                [   float(dout[2][0]) - float(dout[2][2])/2.0,  #xleft
                    float(dout[2][1]) - float(dout[2][3])/2.0,  #ytop
                    float(dout[2][0]) + float(dout[2][2])/2.0,  #xright
                    float(dout[2][1]) + float(dout[2][3])/2.0,  #ybottom
                    float(dout[1])/100.0,                       #confidence
                    self.class_names.index(dout[0]),            #class_id
                ] for dout in darknet_outputs]
            outputs = np.array(outputs)
            #sort outputs by last column
            outputs = outputs[np.argsort(-outputs[:,4])]
        else:
            outputs = None
        return outputs     

def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    trackers = []
    for class_id, class_name in enumerate(predictor.class_names):
        trackers.append(
            BYTETracker(args, frame_rate=args.fps)
        )
    #tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        outputs_total = predictor.predict(img, timer)
        if outputs_total is not None:
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = []

            online_tlwhs_not_active = []
            online_ids_not_active = []
            online_scores_not_active = []
            online_classes_not_active = []


            for tracker_id, tracker in enumerate(trackers):

                # Get rows in outputs total with class_id == tracker_id
                outputs = outputs_total[outputs_total[:, -1].astype(int) == tracker_id]
                online_targets, online_targets_not_active = tracker.update(outputs, [img_height, img_width], [predictor.height, predictor.width])
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_classes.append(t.class_id)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{t.class_id},{int(t.is_activated)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                        )
                for t in online_targets_not_active:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs_not_active.append(tlwh)
                        online_ids_not_active.append(tid)
                        online_scores_not_active.append(t.score)
                        online_classes_not_active.append(t.class_id)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{t.class_id},{int(t.is_activated)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                        )
            timer.toc()

            active_color = [COLOR_GREEN if t.class_id == 0 else COLOR_BLUE for t in online_targets]
            online_im = plot_tracking(
                img, online_tlwhs, online_ids, class_ids=online_classes, colors=active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
            not_active_color = [COLOR_RED]*len(online_tlwhs_not_active)
            online_im = plot_tracking(
                online_im, online_tlwhs_not_active, online_ids_not_active, class_ids=online_classes_not_active, colors=not_active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print video info (width, height, fps, frames)
    logger.info(f"video info: {width} x {height} @ {fps} FPS, {frames} frames")
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split(os.sep)[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    trackers = []

    # We will track different objects with different trackers.
    for class_id, class_name in enumerate(predictor.class_names):
        trackers.append(BYTETracker(args, frame_rate=fps))

    #tracker = BYTETracker(args, frame_rate=30)
    
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if (frame_id + 1) % 20 == 0:
            time_eta = (frames - (frame_id + 1)) * timer.average_time
            time_eta_hhmmss = str(datetime.timedelta(seconds=time_eta))
            logger.info('Processing frame {} ({:.2%}) ({:.2f} fps) ETA: {}'.format(frame_id + 1, (frame_id + 1) / frames, 1. / max(1e-5, timer.average_time), time_eta_hhmmss))
            timer.clear()
        ret_val, frame = cap.read()
        if ret_val:
            
            darknet_raw_img = frame.copy()
            
            outputs_total = predictor.predict(darknet_raw_img, timer)
            
            if outputs_total is not None:
                frame_height, frame_width = frame.shape[0], frame.shape[1]
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_classes = []

                online_tlwhs_not_active = []
                online_ids_not_active = []
                online_scores_not_active = []
                online_classes_not_active = []

                for tracker_id, tracker in enumerate(trackers):
                    # Get rows in outputs total with class_id == tracker_id
                    outputs = outputs_total[outputs_total[:, -1].astype(int) == tracker_id]

                    online_targets, online_targets_not_active = tracker.update(outputs, [frame_height, frame_width], [predictor.height, predictor.width])
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            online_classes.append(t.class_id)
                            results.append(
                                #f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                f"{frame_id},{tid},{t.class_id},{int(t.is_activated)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                            )
                    for t in online_targets_not_active:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs_not_active.append(tlwh)
                            online_ids_not_active.append(tid)
                            online_scores_not_active.append(t.score)
                            online_classes_not_active.append(t.class_id)
                            # save results
                            results.append(
                                f"{frame_id},{tid},{t.class_id},{int(t.is_activated)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                            )

                timer.toc()
                active_color = [COLOR_GREEN if cc == 0 else COLOR_BLUE for cc in online_classes]
                online_im = plot_tracking(
                    darknet_raw_img, online_tlwhs, online_ids, class_ids=online_classes, colors=active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                not_active_color = [COLOR_RED]*len(online_tlwhs_not_active)
                online_im = plot_tracking(
                    online_im, online_tlwhs_not_active, online_ids_not_active, class_ids=online_classes_not_active, colors=not_active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = darknet_raw_img
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(args):
    
    output_dir = osp.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    predictor = Darknet_Predictor(args.config_file, args.data_file, args.weights)
    current_time = time.localtime()
    
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
