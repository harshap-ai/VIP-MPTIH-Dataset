"""
Utility script for VIP-HTD. Uses argument parser to choose functions. 
© Harish Prakash
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import argparse


"""
Bbox visualization (GT)

Note: The relative path of the file should be until the file name only. Eg. '../<file_name>/'
"""


def shift_frame_num(_annot):

    annotations = pd.read_csv(
        _annot,
        names=[
            "frames",
            "playerid",
            "bbox_left_x",
            "bbox_left_y",
            "width",
            "height",
            "confidence",
            "category",
            "visiblity",
        ],
    )

    annotations["confidence"] = 1
    annotations["category"] = 1

    annotations = annotations.sort_values(by=["frames"])

    # Re-indexing frames from 0 to 1
    if annotations["frames"].min() == 0:
        annotations["frames"] += 1

    annotations.to_csv(_annot, header=False, index=False)

    print("\n Done!")


# ------------------------------------------------------------------------------------------------------------------------------------#


def bbox_gt(_annot, _img, flag=0):

    """
    Function to visualize the bounding box annotations for VIP-HTD.

    Inputs: _annot : Ground-truth annotation file.
            _img : Frames directory
            flag : 0 for GT bounding boxes, 1 for superimposing GT bounding boxes onto MOT detections.
    Output: A directory named frame_bboxes containing the visualizations (if flag == 0)
    """

    images = sorted([f for f in os.listdir(_img) if f.endswith(".jpg")])
    annotations = pd.read_csv(
        _annot,
        names=[
            "frames",
            "playerid",
            "bbox_left_x",
            "bbox_left_y",
            "width",
            "height",
            "confidence",
            "category",
            "visiblity",
        ],
    )

    # To convert frames from being 0-indexed to 1-indexed and set {confidence, tracklet_id} = 1 [Optional function]
    shift_frame_num(_annot)

    if flag == 0:
        bbox_dir = os.path.join(os.path.split(_img)[0], "frame_bboxes")

        if not os.path.exists(bbox_dir):
            os.makedirs(bbox_dir)

    # If you need to scale the bounding box (but, not needed for good annotations!)
    w = 1280
    h = 720
    scale = 1  # h/w + 0.05 #Use h/w if image size is not 1280p x 720p

    for i, image in enumerate(images, 1):

        image_path = os.path.join(_img, image)
        image_cur = cv2.imread(image_path)
        subset = annotations.loc[annotations["frames"] == i]

        # Top left Coords
        x1, y1 = subset["bbox_left_x"], subset["bbox_left_y"]
        # Bottom Right Coords
        x4, y4 = x1 + subset["width"] * scale, y1 + subset["height"] * scale

        for j in range(len(x1)):

            color = get_color(subset["playerid"].iloc[j])
            text_position = (int(x4.iloc[j]), int(y1.iloc[j]))

            image_cur = cv2.rectangle(
                image_cur,
                (int(x1.iloc[j]), int(y1.iloc[j])),
                (int(x4.iloc[j]), int(y4.iloc[j])),
                color,
                2,
            )
            image_cur = cv2.putText(
                image_cur,
                str(subset["playerid"].iloc[j]),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if flag == 0:
            cv2.imwrite(
                os.path.join(bbox_dir, os.path.split(image_path)[-1]), image_cur
            )

        else:
            cv2.imwrite(os.path.join(_img, os.path.split(image_path)[-1]), image_cur)

    print("Bounding Boxes for GT Done!!!")


# ------------------------------------------------------------------------------------------------------------------------------------#


def bbox_output(_output, _img, _annot):

    """
    Function to visualize the output bounding box from tracker for VIP-HTD.

    CLI command format: python visualizer.py --action bbox_output --annot_path_gt <> --image_path <>

    Inputs: _annot : Ground-truth annotation file.
            _img : Frames directory
            flag : 0 for GT bounding boxes, 1 for superimposing GT bounding boxes onto MOT detections.
    Output: A directory named frame_bboxes containing the visualizations (if flag == 0)
    """

    images = sorted([f for f in os.listdir(_img) if f.endswith(".jpg")])

    annotations = pd.read_csv(
        _output,
        names=[
            "frames",
            "ped_id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )

    # This creates an 'output_bbox_<name>' folder in the same folder where output annotations are present.
    new_dir = os.path.join(
        os.path.split(_img)[0], "output_bbox_{}".format(os.path.split(_annot)[-1])
    )
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # If you need to scale for bad annotations!
    w = 1280
    h = 720
    scale = 1  # h/w + 0.05 #Use h/w if image size is not 1280p x 720p

    for i, image in enumerate(images, 1):

        image_path = os.path.join(_img, image)
        image = cv2.imread(image_path)

        subset = annotations.loc[annotations["frames"] == i]

        # Top left
        x1, y1 = subset["bb_left"], subset["bb_top"]
        # Bottom Right Coords
        x4, y4 = x1 + subset["bb_width"] * scale, y1 + subset["bb_height"] * scale

        for j in range(len(x1)):

            color = get_color(int(subset["ped_id"].iloc[j]))

            text_position = (int(x4.iloc[j]), int(y1.iloc[j]))
            image_cur = cv2.rectangle(
                image,
                (int(x1.iloc[j]), int(y1.iloc[j])),
                (int(x4.iloc[j]), int(y4.iloc[j])),
                color,
                2,
            )
            image_cur = cv2.putText(
                image_cur,
                str(int(subset["ped_id"].iloc[j])),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(os.path.join(new_dir, os.path.split(image_path)[-1]), image_cur)

    bbox_gt(
        _annot, new_dir, flag=1
    )  # Calling this function to superimpose GT bboxes on top of outputs. Comment this out if you just want the output boxes.

    print("Bounding Boxes for Output Done!!!")


# ------------------------------------------------------------------------------------------------------------------------------------#


def get_color(idx):

    """
    Helper function for bbox_output. Gives a deterministic & different colour for different player IDs.
    """

    idx *= 3
    color = (int((37 * idx) % 255), int((17 * idx) % 255), int((29 * idx) % 255))

    return color


# ------------------------------------------------------------------------------------------------------------------------------------#


def make_video(_path, _fps):

    """
    Function to create a video from Frames (usually to visualize annotations as a video)

    Inputs: _path = Frames Directory path
            _fps = FPS with which we want to output the video.
    Output: A .mp4 video
    """

    images = [frames for frames in sorted(os.listdir(_path)) if frames.endswith(".jpg")]

    image_1 = cv2.imread(os.path.join(_path, images[0]))
    height, width, _ = image_1.shape

    new_dir = os.path.join(
        os.path.split(_path)[0], "video_{}".format(os.path.split(_path)[-2])
    )
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        os.path.join(new_dir, "{}.mp4".format(os.path.split(_path)[1])),
        fourcc,
        _fps,
        (width, height),
    )

    for image in images:

        image_path = os.path.join(_path, image)
        _read = cv2.imread(image_path)

        # Write the current frame to the video
        video_writer.write(_read)

    # Release the video writer and close the video file
    video_writer.release()

    print("Video creation done!!!")


# ------------------------------------------------------------------------------------------------------------------------------------#


def reassign_id(_annot):

    """
    Function to change 'personnel-level' annotations to 'tracklet-level/motchallenge-style annotations'

    Inputs: _annot = Path to GT annotation file (.txt)

    Output: Changed annotation written onto the same file.
    """

    shift_frame_num(_annot)

    annotations = pd.read_csv(
        _annot,
        names=[
            "frames",
            "playerid",
            "bbox_left_x",
            "bbox_left_y",
            "width",
            "height",
            "confidence",
            "tracklet_id",
            "visiblity",
        ],
    )

    unique_player_id = set(annotations.playerid.unique())

    print("The list of unique pre-assigned player ID's are:", unique_player_id)
    last_id = len(unique_player_id)

    update_id = last_id + 1

    for id in unique_player_id:

        frames = annotations.loc[annotations["playerid"] == id]

        frame_diff = frames[
            "frames"
        ].diff()  # Returns the difference between prev (t-1) and next (t) frame #.
        set_indices = frame_diff[frame_diff > 1].index.tolist()
        print("Id:", id, "Set Indices:", set_indices)

        if set_indices:

            frame_sets = []
            start_idx = frames.index[0]

            for end_idx in set_indices:

                frame_set = frames.loc[start_idx : end_idx - 1]
                frame_sets.append(frame_set)
                start_idx = end_idx

            last_set = frames.loc[start_idx:]
            frame_sets.append(last_set)

            if frame_sets:

                for idx, _frame_set in enumerate(frame_sets):

                    if idx != 0:

                        annotations.loc[_frame_set.index, "playerid"] = update_id
                        update_id += 1

    print("The modified Id range is", set(annotations.playerid.unique()))
    annotations.to_csv(_annot, header=False, index=False)

    print(
        "Personnel-level annotations changed to tracklet-level/motchallenge-style annotations"
    )


# ------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-annot_gt",
        "--annot_path_gt",
        action="store",
        default=None,
        type=str,
        help="Relative path to the GT annotation file",
    )
    parser.add_argument(
        "-image",
        "--image_path",
        action="store",
        default=None,
        type=str,
        help="Relative path to the parsed images",
    )
    parser.add_argument(
        "-annot_mot",
        "--annot_path_output",
        action="store",
        default=None,
        type=str,
        help="Relative path to the MOT output annotation file",
    )
    parser.add_argument(
        "-do",
        "--action",
        action="store",
        type=str,
        help="Actions: i. draw bounding box for GT(sanity) - bbox_gt \
                    \n ii. draw bounding box for output (MOT) -  bbox_output \
                    \n iii. Convert Bbox images to video - make_video \
                    \n iv. Reassign frame id - reassign_id",
    )

    parser.add_argument(
        "-f",
        "--fps",
        action="store",
        default=30,
        type=int,
        help="Enter the fps to make video from frames",
    )

    args = parser.parse_args()

    if args.action == "bbox_gt":
        bbox_gt(args.annot_path_gt, args.image_path, flag=0)

    elif args.action == "bbox_output":
        bbox_output(args.annot_path_output, args.image_path, args.annot_path_gt)

    elif args.action == "make_video":
        make_video(args.image_path, args.fps)

    elif args.action == "reassign_id":
        reassign_id(args.annot_path_gt)

    elif args.action == "shift_frame_num":
        shift_frame_num(args.annot_path_gt)

    else:
        print("Please enter a valid action (ref. help of action argument)")


"""

Example terminal command:

python visualizer.py --action bbox_gt --annot_path_gt <> --image_path <> 

"""
