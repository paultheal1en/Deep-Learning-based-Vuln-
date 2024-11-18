static int rm_probe(AVProbeData *p)

{

    /* check file header */

    if (p->buf_size <= 32)

        return 0;

    if ((p->buf[0] == '.' && p->buf[1] == 'R' &&

         p->buf[2] == 'M' && p->buf[3] == 'F' &&

         p->buf[4] == 0 && p->buf[5] == 0) ||

        (p->buf[0] == '.' && p->buf[1] == 'r' &&

         p->buf[2] == 'a' && p->buf[3] == 0xfd))

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
