static void opt_frame_pad_bottom(const char *arg)

{

    frame_padbottom = atoi(arg);

    if (frame_padbottom < 0) {

        fprintf(stderr, "Incorrect bottom pad size\n");

        av_exit(1);

    }

}
