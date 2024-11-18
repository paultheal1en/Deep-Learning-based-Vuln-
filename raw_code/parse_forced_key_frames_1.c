static void parse_forced_key_frames(char *kf, OutputStream *ost,

                                    AVCodecContext *avctx)

{

    char *p;

    int n = 1, i;

    int64_t t;



    for (p = kf; *p; p++)

        if (*p == ',')

            n++;

    ost->forced_kf_count = n;

    ost->forced_kf_pts   = av_malloc(sizeof(*ost->forced_kf_pts) * n);

    if (!ost->forced_kf_pts) {

        av_log(NULL, AV_LOG_FATAL, "Could not allocate forced key frames array.\n");

        exit(1);

    }



    p = kf;

    for (i = 0; i < n; i++) {

        char *next = strchr(p, ',');



        if (next)

            *next++ = 0;



        t = parse_time_or_die("force_key_frames", p, 1);

        ost->forced_kf_pts[i] = av_rescale_q(t, AV_TIME_BASE_Q, avctx->time_base);



        p = next;

    }

}
