import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.utils.visualize import plot_detections, plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


import sys
DARKNET_PATH = "/home/vjpereira/ATV/darknet"
sys.path.append(DARKNET_PATH)
assert os.path.exists(DARKNET_PATH), "Please set the correct path to darknet directory"

import darknet
import numpy as np
import datetime
from shapely.geometry import Polygon
import pandas as pd

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class ATVColor(object):
    BLUE      = (255, 0, 0)
    GREEN     = (0, 255, 0)
    YELLOW    = (0, 255, 255)
    RED       = (0, 0, 255)
    CYAN      = (255, 255, 0)

class ATVTrackType(object):
    active = 0
    inactive = 1
    lost = 2
    detection = 3

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

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
    parser.add_argument("--output_dir", default=".", type=str, help="output dir")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=.5, help="tracking confidence threshold")
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
    parser.add_argument("--thresh", type=float, default=.05, help="remove detections with confidence below this value")
    parser.add_argument("--save_predictions", action="store_true", help="save predictions")
    parser.add_argument("--load_predictions", default=None, help="load predictions dataframe")
    parser.add_argument("--downscale_fps", type=int, default=1, help="downscale output fps")
    parser.add_argument("--resize_output", action="store_true", help="resize output")
    parser.add_argument("--resize_output_width", type=int, default=1280, help="resize output width")
    parser.add_argument("--resize_output_height", type=int, default=720, help="resize output height")
    parser.add_argument("--filter_labels", nargs="+", default=[], choices=["ball", "player"], help="filter by specific labels") 
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
        roi=None,
        batch_size=1,
        filter_label=[]
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
        # roi_points is a 2d array. rows are pixel coordinates and columns are x,y coordinates. We create a polygon from these points.
        self.roi = roi
        self.filter_label = filter_label
        
    def predict(self, frame, timer, frame_id=None):
        
        raw_img = frame.copy()
        raw_img_height, raw_img_width = raw_img.shape[:2]
        frame_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.width, self.height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
    
        timer.tic()
        darknet_outputs = darknet.detect_image(self.network, self.class_names, img_for_detect, self.threshold)
        darknet.free_image(img_for_detect)

        if len(self.filter_label) > 0:
            darknet_outputs = [out for out in darknet_outputs if out[0] in self.filter_label]

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
            outputs = self.check_roi(outputs,[raw_img_width,raw_img_height])
            # Saturate xleft to 0, ytop to 0, xright to self.width, ybottom to self.height
            outputs[:, 0] = np.clip(outputs[:, 0], 0, self.width)
            outputs[:, 1] = np.clip(outputs[:, 1], 0, self.height)
            outputs[:, 2] = np.clip(outputs[:, 2], 0, self.width)
            outputs[:, 3] = np.clip(outputs[:, 3], 0, self.height)
        else:
            outputs = None
        return outputs     

    #define function to check if detections are inside of roi
    def check_roi(self, detections, img_size):
        if self.roi is None:
            return detections
        else:
            # Resize roi to match network size bounding boxes
            [img_w,img_h] = img_size
            roi_resized = self.roi * np.array([self.width/img_w, self.height/img_h])
            roi_poly = Polygon(roi_resized)
            in_roi = []
            for i in range(detections.shape[0]):
                bbox_polygon = Polygon(
                    [
                        (detections[i,0], detections[i,1]),
                        (detections[i,2], detections[i,1]),
                        (detections[i,2], detections[i,3]),
                        (detections[i,0], detections[i,3]),
                    ]
                )
                if bbox_polygon.intersects(roi_poly):
                    in_roi.append(i)
            return detections[in_roi,:] if len(in_roi) > 0 else None

    def detections_to_pixel(self, detections, img_size):
        [img_h,img_w] = img_size
        detections_pixel = detections.copy()
        detections_pixel[:,0] = detections[:,0] * (img_w/self.width)
        detections_pixel[:,1] = detections[:,1] * (img_h/self.height)
        detections_pixel[:,2] = detections[:,2] * (img_w/self.width)
        detections_pixel[:,3] = detections[:,3] * (img_h/self.height)
        return detections_pixel

    def tlbr_to_tlwh(self, detections):
        detections_tlwh = detections.copy()
        detections_tlwh[:,0] = detections[:,0] - detections[:,2]/2.0
        detections_tlwh[:,1] = detections[:,1] - detections[:,3]/2.0
        detections_tlwh[:,2] = detections[:,2]
        detections_tlwh[:,3] = detections[:,3]
        return detections_tlwh

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

            online_tlwhs_lost = []
            online_ids_lost = []
            online_scores_lost = []
            online_classes_lost = []


            for tracker_id, tracker in enumerate(trackers):

                # Get rows in outputs total with class_id == tracker_id
                outputs = outputs_total[outputs_total[:, -1].astype(int) == tracker_id]
                online_targets, online_targets_not_active, online_targets_lost = tracker.update(outputs, [img_height, img_width], [predictor.height, predictor.width])
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
                for t in online_targets_lost:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs_lost.append(tlwh)
                        online_ids_lost.append(tid)
                        online_scores_lost.append(t.score)
                        online_classes_lost.append(t.class_id)
                        # save results
                        #results.append(
                        #    f"{frame_id},{tid},{t.class_id},{int(t.is_activated)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                        #)
            timer.toc()


            active_color = [ATVColor.GREEN if t.class_id == 0 else ATVColor.BLUE for t in online_targets]
            online_im = plot_tracking(
                img, online_tlwhs, online_ids, class_ids=online_classes, colors=active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
            not_active_color = [ATVColor.YELLOW]*len(online_tlwhs_not_active)
            online_im = plot_tracking(
                online_im, online_tlwhs_not_active, online_ids_not_active, class_ids=online_classes_not_active, colors=not_active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
            lost_color = [ATVColor.RED]*len(online_tlwhs_lost)
            online_im = plot_tracking(
                online_im, online_tlwhs_lost, online_ids_lost, class_ids=online_classes_lost, colors=lost_color, frame_id=frame_id + 1, fps=1. / timer.average_time
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
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    output_width, output_height = width, height
    if args.resize_output:
        output_width, output_height = args.resize_output_width, args.resize_output_height
    
    output_fps = fps / args.downscale_fps

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (output_width, output_height)
    )
    
    trackers = []
    # We will track different objects with different trackers.
    for class_id, class_name in enumerate(predictor.class_names):
        trackers.append(BYTETracker(args, frame_rate=output_fps))
        #break # Do only class 0: ball
    #tracker = BYTETracker(args, frame_rate=30)
    
    timer = Timer()
    timer_general = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % args.downscale_fps > 0:
            _, _ = cap.read()
            frame_id += 1
            continue
        if frame_id % 20 == 0:
            time_eta = (frames - (frame_id + 1)) * timer_general.average_time
            timer_general.clear()
            time_eta_hhmmss = str(datetime.timedelta(seconds=time_eta))
            logger.info('Processing frame {} ({:.2%}) ({:.2f} fps) ETA: {}'.format(frame_id, (frame_id + 1) / frames, 1. / max(1e-5, timer.average_time), time_eta_hhmmss))
            #timer.clear()
        timer_general.tic()
        ret_val, frame = cap.read()
        if ret_val:
            
            if args.resize_output:
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            darknet_raw_img = frame.copy()
            
            outputs_total = predictor.predict(darknet_raw_img, timer)
            
            if outputs_total is not None:
                
                frame_height, frame_width = frame.shape[0], frame.shape[1]
                outputs_total_resized = predictor.detections_to_pixel(outputs_total,[frame_height, frame_width])
                outputs_total_resized_tlwh = predictor.tlbr_to_tlwh(outputs_total_resized)
                for output_detected in outputs_total_resized_tlwh:
                    results.append(
                        f"{frame_id},-1,{ predictor.class_names[int(output_detected[5])]},{ATVTrackType.detection},{output_detected[0]:.2f},{output_detected[1]:.2f},{output_detected[2]:.2f},{output_detected[3]:.2f},{output_detected[4]:.2f}\n"
                    )

                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_classes = []

                online_tlwhs_not_active = []
                online_ids_not_active = []
                online_scores_not_active = []
                online_classes_not_active = []

                online_tlwhs_lost = []
                online_ids_lost = []
                online_scores_lost = []
                online_classes_lost = []

                for tracker_id, tracker in enumerate(trackers):
                    # Get rows in outputs total with class_id == tracker_id
                    outputs = outputs_total[outputs_total[:, -1].astype(int) == tracker_id]
                    online_targets, online_targets_not_active, online_targets_lost = tracker.update(outputs, [frame_height, frame_width], [predictor.height, predictor.width])
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
                                f"{frame_id},{tid},{predictor.class_names[int(t.class_id)]},{int(ATVTrackType.active)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
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
                                f"{frame_id},{tid},{predictor.class_names[int(t.class_id)]},{int(ATVTrackType.inactive)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                            )
                    for t in online_targets_lost:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs_lost.append(tlwh)
                            online_ids_lost.append(tid)
                            online_scores_lost.append(t.score)
                            online_classes_lost.append(t.class_id)
                            # save results
                            results.append(
                                f"{frame_id},{tid},{predictor.class_names[int(t.class_id)]},{int(ATVTrackType.lost)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}\n"
                            )
                    #break # Do only class 0: ball
                timer.toc()
                active_color = [ATVColor.GREEN if cc == 0 else ATVColor.BLUE for cc in online_classes]
                online_im = plot_tracking(
                    darknet_raw_img, online_tlwhs, online_ids, class_ids=online_classes, colors=active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                not_active_color = [ATVColor.YELLOW]*len(online_tlwhs_not_active)
                online_im = plot_tracking(
                    online_im, online_tlwhs_not_active, online_ids_not_active, class_ids=online_classes_not_active, colors=not_active_color, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                lost_color = [ATVColor.RED]*len(online_tlwhs_lost)
                text_info = {'frame': frame_id, 'fps': f"{1. / timer.average_time:.2f}", 'active': len(online_tlwhs), 'not_active': len(online_tlwhs_not_active), 'lost': len(online_tlwhs_lost), 'detections': len(outputs_total)}
                online_im = plot_tracking(
                    online_im, online_tlwhs_lost, online_ids_lost, class_ids=online_classes_lost, colors=lost_color, frame_id=frame_id + 1, fps=1. / timer.average_time, info=text_info
                )
                if len(outputs_total) > 0:
                    detections_color = [ATVColor.CYAN]*len(outputs_total)
                    online_im = plot_detections(online_im, outputs_total_resized, colors=detections_color, fill=True, alpha=.5)
            else:
                timer.toc()
                text_info = {'frame': frame_id, 'fps': f"{1. / timer.average_time:.2f}", 'active': ' ', 'not_active': ' ', 'lost': ' ', 'detections': 0}
                online_im = plot_tracking(
                    darknet_raw_img, [], [], class_ids=[], colors=[], frame_id=frame_id + 1, fps=1. / timer.average_time, info = text_info
                )
                #online_im = darknet_raw_img
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
        timer_general.toc()

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

    if False:
        roi = np.array([[0,225],[1897,556],[3792,1328],[0,2004]])
    else:
        roi = None
    
    predictor = Darknet_Predictor(args.config_file, args.data_file, args.weights, thresh=args.thresh, roi=roi, filter_label=args.filter_labels)
    current_time = time.localtime()
    
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
