import av
import sys

# Replace this with the VM path if you copy this script over!
# /mnt/dataset-swan/data/Schwan_T3_FineTune/MAJARO01_MUC_T3/chunks/MAJARO01_MUC_T3_Caregiver_Engagement_idx0029_Cpos_kamera2.mp4
video_path = r"X:\data\Schwan_T3_FineTune\MAJARO01_MUC_T3\chunks\MAJARO01_MUC_T3_Caregiver_Engagement_idx0029_Cpos_kamera2.mp4"

def test_video(path):
    print(f"Testing video:\n  {path}\n")
    try:
        # This is the exact function LLaMA Factory calls to decode videos
        with av.open(path, "r") as container:
            print("✅ SUCCESS: PyAV successfully opened and decoded the video headers.")
    except Exception as e:
        print("❌ CORRUPT DETECTED:")
        print(f"   PyAV explicitly threw an error: {e}")
        print("   -> This video will now be filtered out by build_manifest.py!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    test_video(video_path)
