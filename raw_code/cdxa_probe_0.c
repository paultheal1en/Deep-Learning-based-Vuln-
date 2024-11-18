static int cdxa_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf[0] == 'R' && p->buf[1] == 'I' &&

        p->buf[2] == 'F' && p->buf[3] == 'F' &&

        p->buf[8] == 'C' && p->buf[9] == 'D' &&

        p->buf[10] == 'X' && p->buf[11] == 'A')

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
