static void opt_qsquish(const char *arg)

{

    video_qsquish = atof(arg);

    if (video_qsquish < 0.0 ||

        video_qsquish > 99.0) {

        fprintf(stderr, "qsquish must be >= 0.0 and <= 99.0\n");

        exit(1);

    }

}
