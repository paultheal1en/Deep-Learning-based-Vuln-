static void opt_audio_sample_fmt(const char *arg)

{

    if (strcmp(arg, "list"))

        audio_sample_fmt = av_get_sample_fmt(arg);

    else {

        list_fmts(av_get_sample_fmt_string, AV_SAMPLE_FMT_NB);

        ffmpeg_exit(0);

    }

}
