from argparse import ArgumentParser

class cli_parser:

    def parse():
        parser = ArgumentParser(
            prog="VisionAI",
            description="VisionAI is a environment for annotation, train, and inference for vision AIs."
        )


        # Live mode
        subparsers = parser.add_subparsers(dest="run_mode", help="run mode", required=True)
        live_parser = subparsers.add_parser("live", help="live mode")
        # parser.add_argument("run_mode", type=str, help="run mode", default="live", choices="live, test, train".split(","))
        live_parser.add_argument("-nv", "--no_video", action="store_true", help="disable video")
        live_parser.add_argument("-cap", "--capture_objects", action="store_true", help="capture objects")
        live_parser.add_argument("-pl", "--performance_log", action="store_true", help="performance log")
        live_parser.add_argument("-rtsp", "--rtsp", action="store_true", help="rtsp")
        live_parser.add_argument("-test", "--test", action="store_true", help="test")
        live_parser.add_argument("-sf", "--skip_frames", type=int, help="skip frames", default=0)
        live_parser.add_argument("-rec", "--record", type=str, help="record", default="")
        live_parser.add_argument("-f", "--file", type=str, help="file")

        # Train mode
        train_parser = subparsers.add_parser("train", help="train mode")

        # Test mode
        test_parser = subparsers.add_parser("test", help="test mode")

        annotate_parser = subparsers.add_parser("annotate", help="annotate mode")
        annotate_parser.add_argument("-p", "--path", type=str, help="folder to images to annotate")
        annotate_parser.add_argument("-demo", "--demo", action="store_true", help="annotation demo")

        args = parser.parse_args()

        return args