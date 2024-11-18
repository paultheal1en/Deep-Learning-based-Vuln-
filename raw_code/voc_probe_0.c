static int voc_probe(AVProbeData *p)

{

    int version, check;



    if (p->buf_size < 26)

        return 0;

    if (memcmp(p->buf, voc_magic, sizeof(voc_magic) - 1))

        return 0;

    version = p->buf[22] | (p->buf[23] << 8);

    check = p->buf[24] | (p->buf[25] << 8);

    if (~version + 0x1234 != check)

        return 10;



    return AVPROBE_SCORE_MAX;

}
