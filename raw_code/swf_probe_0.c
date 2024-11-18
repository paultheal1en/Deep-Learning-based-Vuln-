static int swf_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf_size <= 16)

        return 0;

    if ((p->buf[0] == 'F' || p->buf[0] == 'C') && p->buf[1] == 'W' &&

        p->buf[2] == 'S')

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
