static int tta_probe(AVProbeData *p)

{

    const uint8_t *d = p->buf;

    if (p->buf_size < 4)

        return 0;

    if (d[0] == 'T' && d[1] == 'T' && d[2] == 'A' && d[3] == '1')

        return 80;

    return 0;

}
