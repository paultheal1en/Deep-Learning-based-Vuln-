static int cin_probe(AVProbeData *p)

{

    if (p->buf_size < 18)

        return 0;



    /* header starts with this special marker */

    if (AV_RL32(&p->buf[0]) != 0x55AA0000)

        return 0;



    /* for accuracy, check some header field values */

    if (AV_RL32(&p->buf[12]) != 22050 || p->buf[16] != 16 || p->buf[17] != 0)

        return 0;



    return AVPROBE_SCORE_MAX;

}
