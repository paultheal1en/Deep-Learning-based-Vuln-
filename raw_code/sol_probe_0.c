static int sol_probe(AVProbeData *p)

{

    /* check file header */

    uint16_t magic;

    if (p->buf_size <= 14)

        return 0;

    magic=le2me_16(*((uint16_t*)p->buf));

    if ((magic == 0x0B8D || magic == 0x0C0D || magic == 0x0C8D) &&

        p->buf[2] == 'S' && p->buf[3] == 'O' &&

        p->buf[4] == 'L' && p->buf[5] == 0)

        return AVPROBE_SCORE_MAX;

    else

        return 0;

}
