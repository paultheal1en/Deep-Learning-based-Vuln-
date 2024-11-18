static int aiff_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf_size < 16)

        return 0;

    if (p->buf[0] == 'F' && p->buf[1] == 'O' &&

        p->buf[2] == 'R' && p->buf[3] == 'M' &&

        p->buf[8] == 'A' && p->buf[9] == 'I' &&

        p->buf[10] == 'F' && (p->buf[11] == 'F' || p->buf[11] == 'C'))

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
