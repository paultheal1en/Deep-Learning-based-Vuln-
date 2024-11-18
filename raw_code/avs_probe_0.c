static int avs_probe(AVProbeData * p)

{

    const uint8_t *d;



    if (p->buf_size < 2)

        return 0;

    d = p->buf;

    if (d[0] == 'w' && d[1] == 'W' && d[2] == 0x10 && d[3] == 0)

        return 50;



    return 0;

}
