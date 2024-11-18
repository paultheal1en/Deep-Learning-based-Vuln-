static void opt_frame_size(const char *arg)

{

    if (av_parse_video_frame_size(&frame_width, &frame_height, arg) < 0) {

        fprintf(stderr, "Incorrect frame size\n");

        av_exit(1);

    }

    if ((frame_width % 2) != 0 || (frame_height % 2) != 0) {

        fprintf(stderr, "Frame size must be a multiple of 2\n");

        av_exit(1);

    }

}
