
mkdir calibrationImages

# -r[:stream_specifier] fps (input/output,per-stream)
#  Set frame rate (Hz value, fraction or abbreviation).
#  -r 1 to get each frame as an image
# -f fmt (input/output)
#  Force input or output file format. The format is normally auto detected for input files and guessed from the file extension for output files, so this option is not needed in most cases.
#  -f image2 to extract images from a video
ffmpeg -i calibrationVideo.MP4 -r 2 -f image2 ./calibrationImages/image-%03d.jpeg
