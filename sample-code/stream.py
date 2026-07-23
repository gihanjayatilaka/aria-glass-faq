import argparse
import time

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

from projectaria_tools.core.mps import EyeGaze, hand_tracking, OpenLoopTrajectoryPose
from projectaria_tools.core.sensor_data import (
    AudioData,
    AudioDataRecord,
    FrontendOutput,
    ImageData,
    ImageDataRecord,
    MotionData,
)

# Set up the device client to initiate connection to the device
device_client = sdk_gen2.DeviceClient()


def device_streaming():
    # Set up the device client config to specify the device to be connected to e.g. device serial number.
    # If nothing is specified, the first device in the list of connected devices will be connected to
    config = sdk_gen2.DeviceClientConfig()
    device_client.set_client_config(config)
    device = device_client.connect()

    # Set recording config with profile name
    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = "profile9"
    device.set_streaming_config(streaming_config)

    # Start and stop recording
    device.start_streaming()
    return device


def image_callback(image_data: ImageData, image_record: ImageDataRecord):
    print(
        f"Received image data of size {image_data.to_numpy_array().shape} with timestamp {image_record.capture_timestamp_ns} ns"
    )


def audio_callback(
    audio_data: AudioData, audio_record: AudioDataRecord, num_channels: int
):
    print(
        f"Received audio data with {len(audio_data.data)} samples and {len(audio_record.capture_timestamps_ns)} timestamps and num channels {num_channels}"
    )


def imu_callback(imu_data: MotionData, sensor_label: str):
    print(
        f"Received {sensor_label} accel data {imu_data.accel_msec2} and gyro {imu_data.gyro_radsec}"
    )


def eyegaze_callback(eyegaze_data: EyeGaze):
    print(
        f"Received EyeGaze data at timestamp {eyegaze_data.tracking_timestamp.total_seconds()} sec "
        f"with yaw={eyegaze_data.yaw:.3f} rad, pitch={eyegaze_data.pitch:.3f} rad, "
        f"depth={eyegaze_data.depth:.3f} m"
    )


def handtracking_callback(handtracking_data: hand_tracking.HandTrackingResult):
    print(
        f"Received HandTracking data at timestamp {handtracking_data.tracking_timestamp.total_seconds()} sec"
    )

    # Check left hand data
    if handtracking_data.left_hand is not None:
        left_hand = handtracking_data.left_hand
        print(f"  Left hand confidence: {left_hand.confidence:.3f}")
        print(f"  Left wrist position: {left_hand.get_wrist_position_device()}")
        print(f"  Left palm position: {left_hand.get_palm_position_device()}")
        if left_hand.wrist_and_palm_normal_device is not None:
            normals = left_hand.wrist_and_palm_normal_device
            print(f"  Left wrist normal: {normals.wrist_normal_device}")
            print(f"  Left palm normal: {normals.palm_normal_device}")
    else:
        print("  Left hand: No data")

    # Check right hand data
    if handtracking_data.right_hand is not None:
        right_hand = handtracking_data.right_hand
        print(f"  Right hand confidence: {right_hand.confidence:.3f}")
        print(f"  Right wrist position: {right_hand.get_wrist_position_device()}")
        print(f"  Right palm position: {right_hand.get_palm_position_device()}")
        if right_hand.wrist_and_palm_normal_device is not None:
            normals = right_hand.wrist_and_palm_normal_device
            print(f"  Right wrist normal: {normals.wrist_normal_device}")
            print(f"  Right palm normal: {normals.palm_normal_device}")
    else:
        print("  Right hand: No data")


def vio_callback(vio_data: FrontendOutput):
    print(
        f"Received VIO data at timestamp {vio_data.capture_timestamp_ns} with transform_odometry_bodyimu: {vio_data.transform_odometry_bodyimu.rotation().log()} and {vio_data.transform_odometry_bodyimu.translation()} ns"
    )


def calib_callback(calib_json_str: str):
    print(f"Received calibration: {calib_json_str}")


def setup_streaming_receiver(device, record_to_vrs):
    # setup the server to receive streaming data from the device
    # IP address : 0.0.0.0 means that the server is listening on all available interfaces
    # Port : 6768 is the port number that the server is listening on
    config = sdk_gen2.HttpServerConfig()
    config.address = "0.0.0.0"
    config.port = 6768

    # setup the receiver
    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(config)
    if record_to_vrs != "":
        stream_receiver.record_to_vrs(record_to_vrs)

    # register callbacks for each type of data
    stream_receiver.register_slam_callback(image_callback)
    stream_receiver.register_rgb_callback(image_callback)
    stream_receiver.register_audio_callback(audio_callback)
    stream_receiver.register_eye_gaze_callback(eyegaze_callback)
    stream_receiver.register_hand_pose_callback(handtracking_callback)
    stream_receiver.register_vio_callback(vio_callback)

    # start the server
    stream_receiver.start_server()

    time.sleep(10)

    # stop streaming and terminate the server
    device.stop_streaming()

    time.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record-to-vrs",
        dest="record_to_vrs",
        type=str,
        default="/home/gihan/aria/streamed",
        required=False,
        help="Output directory to save the received streaming into VRS",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # setup device to start streaming
    device = device_streaming()

    # setup streaming receiver to receive streaming data with callbacks
    setup_streaming_receiver(device, args.record_to_vrs)
