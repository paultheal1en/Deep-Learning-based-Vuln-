static int h261_probe(AVProbeData *p)

{

    int code;

    const uint8_t *d;



    if (p->buf_size < 6)

        return 0;

    d = p->buf;

    code = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4);

    if (code == 0x10) {

        return 50;

    }

    return 0;

}
