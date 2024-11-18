static int au_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf_size <= 24)

        return 0;

    if (p->buf[0] == '.' && p->buf[1] == 's' &&

        p->buf[2] == 'n' && p->buf[3] == 'd')

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
