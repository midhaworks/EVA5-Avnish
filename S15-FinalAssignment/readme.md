

Use ffmpeg to extract images from interior videos every 1 sec:
ffmpeg -i DESIGNERVILLA.mp4 -r 1 ../interiors/int%04d.png

