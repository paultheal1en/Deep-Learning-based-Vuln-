static void opt_frame_pad_top(const char *arg)

{

    frame_padtop = atoi(arg);

    if (frame_padtop < 0) {

        fprintf(stderr, "Incorrect top pad size\n");

        av_exit(1);

    }

}
