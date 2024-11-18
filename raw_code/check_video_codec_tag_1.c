static int check_video_codec_tag(int codec_tag) {

    if (codec_tag <= 0 || codec_tag > 15) {

        return AVERROR(ENOSYS);

    } else

        return 0;

}
