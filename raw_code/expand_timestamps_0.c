static void expand_timestamps(void *log, struct sbg_script *s)

{

    int i, nb_rel = 0;

    int64_t now, cur_ts, delta = 0;



    for (i = 0; i < s->nb_tseq; i++)

        nb_rel += s->tseq[i].ts.type == 'N';

    if (nb_rel == s->nb_tseq) {

        /* All ts are relative to NOW: consider NOW = 0 */

        now = 0;

        if (s->start_ts != AV_NOPTS_VALUE)

            av_log(log, AV_LOG_WARNING,

                   "Start time ignored in a purely relative script.\n");

    } else if (nb_rel == 0 && s->start_ts != AV_NOPTS_VALUE ||

               s->opt_start_at_first) {

        /* All ts are absolute and start time is specified */

        if (s->start_ts == AV_NOPTS_VALUE)

            s->start_ts = s->tseq[0].ts.t;

        now = s->start_ts;

    } else {

        /* Mixed relative/absolute ts: expand */

        time_t now0;

        struct tm *tm, tmpbuf;



        av_log(log, AV_LOG_WARNING,

               "Scripts with mixed absolute and relative timestamps can give "

               "unexpected results (pause, seeking, time zone change).\n");

#undef time

        time(&now0);

        tm = localtime_r(&now0, &tmpbuf);

        now = tm ? tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec :

                   now0 % DAY;

        av_log(log, AV_LOG_INFO, "Using %02d:%02d:%02d as NOW.\n",

               (int)(now / 3600), (int)(now / 60) % 60, (int)now % 60);

        now *= AV_TIME_BASE;

        for (i = 0; i < s->nb_tseq; i++) {

            if (s->tseq[i].ts.type == 'N') {

                s->tseq[i].ts.t += now;

                s->tseq[i].ts.type = 'T'; /* not necessary */

            }

        }

    }

    if (s->start_ts == AV_NOPTS_VALUE)

        s->start_ts = s->opt_start_at_first ? s->tseq[0].ts.t : now;

    s->end_ts = s->opt_duration ? s->start_ts + s->opt_duration :

                AV_NOPTS_VALUE; /* may be overridden later by -E option */

    cur_ts = now;

    for (i = 0; i < s->nb_tseq; i++) {

        if (s->tseq[i].ts.t + delta < cur_ts)

            delta += DAY_TS;

        cur_ts = s->tseq[i].ts.t += delta;

    }

}
