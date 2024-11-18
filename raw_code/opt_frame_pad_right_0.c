static void opt_frame_pad_right(const char *arg)

{

    frame_padright = atoi(arg);

    if (frame_padright < 0) {

        fprintf(stderr, "Incorrect right pad size\n");

        av_exit(1);

    }

}
