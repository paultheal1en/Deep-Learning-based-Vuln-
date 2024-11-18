static void print_report(const char *op, struct timeval *t, int64_t offset,

                         int64_t count, int64_t total, int cnt, int Cflag)

{

    char s1[64], s2[64], ts[64];



    timestr(t, ts, sizeof(ts), Cflag ? VERBOSE_FIXED_TIME : 0);

    if (!Cflag) {

        cvtstr((double)total, s1, sizeof(s1));

        cvtstr(tdiv((double)total, *t), s2, sizeof(s2));

        printf("%s %"PRId64"/%"PRId64" bytes at offset %" PRId64 "\n",

               op, total, count, offset);

        printf("%s, %d ops; %s (%s/sec and %.4f ops/sec)\n",

               s1, cnt, ts, s2, tdiv((double)cnt, *t));

    } else {/* bytes,ops,time,bytes/sec,ops/sec */

        printf("%"PRId64",%d,%s,%.3f,%.3f\n",

            total, cnt, ts,

            tdiv((double)total, *t),

            tdiv((double)cnt, *t));

    }

}
