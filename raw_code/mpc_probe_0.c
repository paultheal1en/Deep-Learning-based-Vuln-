static int mpc_probe(AVProbeData *p)

{

    const uint8_t *d = p->buf;

    if (p->buf_size < 32)

        return 0;

    if (d[0] == 'M' && d[1] == 'P' && d[2] == '+' && (d[3] == 0x17 || d[3] == 0x7))

        return AVPROBE_SCORE_MAX;

    if (d[0] == 'I' && d[1] == 'D' && d[2] == '3')

        return AVPROBE_SCORE_MAX / 2;

    return 0;

}
