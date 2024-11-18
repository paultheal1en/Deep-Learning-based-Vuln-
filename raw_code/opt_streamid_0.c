static int opt_streamid(const char *opt, const char *arg)

{

    int idx;

    char *p;

    char idx_str[16];



    av_strlcpy(idx_str, arg, sizeof(idx_str));

    p = strchr(idx_str, ':');

    if (!p) {

        fprintf(stderr,

                "Invalid value '%s' for option '%s', required syntax is 'index:value'\n",

                arg, opt);

        ffmpeg_exit(1);

    }

    *p++ = '\0';

    idx = parse_number_or_die(opt, idx_str, OPT_INT, 0, INT_MAX);

    streamid_map = grow_array(streamid_map, sizeof(*streamid_map), &nb_streamid_map, idx+1);

    streamid_map[idx] = parse_number_or_die(opt, p, OPT_INT, 0, INT_MAX);

    return 0;

}
