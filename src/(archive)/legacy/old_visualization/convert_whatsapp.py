import os
import glob
from moviepy.video.io.VideoFileClip import VideoFileClip

VID_DIR = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\metadata\videos"

def convert_videos():
    vid_files = glob.glob(os.path.join(VID_DIR, "advio-*.mp4"))
    
    for vf in vid_files:
        if "whatsapp" in vf:
            continue
            
        out_f = vf.replace(".mp4", "_whatsapp.mp4")
        print(f"Converting {os.path.basename(vf)} for WhatsApp compatibility...")
        print(f" -> Output: {os.path.basename(out_f)}")
        
        try:
            # We explicitly ask moviepy to write out using libx264 & aac which WhatsApp supports natively.
            clip = VideoFileClip(vf)
            clip.write_videofile(
                out_f, 
                codec="libx264", 
                audio_codec="aac", 
                preset="medium",     # balances size and speed
                logger="bar"
            )
            clip.close()
            print(f"Conversion successful: {out_f}\n")
        except Exception as e:
            print(f"Failed to convert {vf}: {e}\n")

if __name__ == "__main__":
    convert_videos()
