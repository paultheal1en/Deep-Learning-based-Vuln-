static int opt_input_ts_scale(const char *opt, const char *arg)

{

    unsigned int stream;

    double scale;

    char *p;



    stream = strtol(arg, &p, 0);

    if (*p)

        p++;

    scale= strtod(p, &p);



    if(stream >= MAX_STREAMS)

        ffmpeg_exit(1);



    ts_scale = grow_array(ts_scale, sizeof(*ts_scale), &nb_ts_scale, stream + 1);

    ts_scale[stream] = scale;

    return 0;

}
