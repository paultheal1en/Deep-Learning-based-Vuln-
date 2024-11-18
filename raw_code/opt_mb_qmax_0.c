static void opt_mb_qmax(const char *arg)

{

    video_mb_qmax = atoi(arg);

    if (video_mb_qmax < 0 ||

        video_mb_qmax > 31) {

        fprintf(stderr, "qmax must be >= 1 and <= 31\n");

        exit(1);

    }

}
