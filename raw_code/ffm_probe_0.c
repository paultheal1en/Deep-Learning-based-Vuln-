static int ffm_probe(AVProbeData *p)

{

    if (p->buf_size >= 4 &&

        p->buf[0] == 'F' && p->buf[1] == 'F' && p->buf[2] == 'M' &&

        p->buf[3] == '1')

        return AVPROBE_SCORE_MAX + 1;

    return 0;

}
