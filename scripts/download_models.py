from pathlib import Path                  # builds file paths
from urllib.request import urlretrieve    # downloads a URL to a file

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {                                # file name -> where to download it from
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "blaze_face_short_range.tflite": f"{BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def main():
    out_dir = Path(__file__).parent.parent / "models"   # scripts/ -> project root -> models/
    out_dir.mkdir(exist_ok=True)                        # create models/ if it's missing
    for name, url in MODELS.items():                    # loop over each model
        target = out_dir / name
        if target.exists():                             # don't download twice
            print(f"skip {name} (already there)")
            continue
        print(f"downloading {name} ...")
        urlretrieve(url, target)
    print("done")


if __name__ == "__main__":
    main()