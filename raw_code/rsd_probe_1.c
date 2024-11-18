static int rsd_probe(AVProbeData *p)

{

    if (!memcmp(p->buf, "RSD", 3) &&

        p->buf[3] - '0' >= 2 && p->buf[3] - '0' <= 6)

        return AVPROBE_SCORE_EXTENSION;

    return 0;

}
