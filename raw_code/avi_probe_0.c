static int avi_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf_size <= 32)

        return 0;

    if (p->buf[0] == 'R' && p->buf[1] == 'I' &&

        p->buf[2] == 'F' && p->buf[3] == 'F' &&

        p->buf[8] == 'A' && p->buf[9] == 'V' &&

        p->buf[10] == 'I' && (p->buf[11] == ' ' || p->buf[11] == 0x19))

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
