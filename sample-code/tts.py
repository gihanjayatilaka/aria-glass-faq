import argparse
import aria.sdk_gen2 as sdk_gen2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        dest="text",
        type=str,
        default="",
        required=True,
        help="TTS text to rendered by the device.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set up the device client to initiate connection to the device
    device_client = sdk_gen2.DeviceClient()
    # Set up the device client config to specify the device to be connected to e.g. device serial number.
    # If nothing is specified, the first device in the list of connected devices will be connected to
    config = sdk_gen2.DeviceClientConfig()

    device_client.set_client_config(config)
    device = device_client.connect()
    print(f"Connected to device: {device.connection_id()}")
    print(f"Rendering TTS: {args.text}")
    device.render_tts(text=args.text)
