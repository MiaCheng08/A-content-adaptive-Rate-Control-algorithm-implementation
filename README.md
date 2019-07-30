# A content adaptive Rate Control solution
To explore the Content Adaptive Encoding, I try to implement  a GOP-level rate Control solution

*Usage*

Please keep all the files in the same path, and run the "x265.exe" with changed/specific command:
x265 --preset 5 --tune psnr --cae --crf [23] --targetR [TARGETR] --input [YOUTPATH] --fps [FPS] --input-res [RES] --output [YOUTPATH] --min-keyint 40 --rc-lookahead 100 --feature [YOUTPATH] --python [YOUPYTHONTPATH] --CAEmodel [THECallByCodecTPATH]

*Explanation*

--cae: enable our self-designed rate control method (with a valid "--feature") 
--crf: unimportant but must
--targetR: specify the target bitrate in kbps

--bitrate: enable single-pass ABR rate control which belong to orignal x265
--qp: enable CQP rate control which belong to orignal x265


*More Explanation*

Resolutions of videos in dataset we train on are 720p(720x1280 or 1280x720) and their frame rates are no more than 30fps (a few videos exceed 0.5 roughly)
Therefore, performance will be better if you use videos whose resolution are 720p and frame rate are below 30fps : )
