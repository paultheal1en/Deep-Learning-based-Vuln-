static int libquvi_probe(AVProbeData *p)

{

    int score;

    quvi_t q;

    QUVIcode rc;



    rc = quvi_init(&q);

    if (rc != QUVI_OK)

        return AVERROR(ENOMEM);

    score = quvi_supported(q, (char *)p->filename) == QUVI_OK ? AVPROBE_SCORE_EXTENSION : 0;

    quvi_close(&q);

    return score;

}
