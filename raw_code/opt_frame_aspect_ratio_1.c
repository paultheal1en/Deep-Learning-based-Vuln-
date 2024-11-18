static void opt_frame_aspect_ratio(const char *arg)

{

    int x = 0, y = 0;

    double ar = 0;

    const char *p;

    char *end;



    p = strchr(arg, ':');

    if (p) {

        x = strtol(arg, &end, 10);

        if (end == p)

            y = strtol(end+1, &end, 10);

        if (x > 0 && y > 0)

            ar = (double)x / (double)y;

    } else

        ar = strtod(arg, NULL);



    if (!ar) {

        fprintf(stderr, "Incorrect aspect ratio specification.\n");

        ffmpeg_exit(1);

    }

    frame_aspect_ratio = ar;



    x = vfilters ? strlen(vfilters) : 0;

    vfilters = av_realloc(vfilters, x+100);

    snprintf(vfilters+x, x+100, "%csetdar=%f\n", x?',':' ', ar);

}
