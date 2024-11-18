static int vivo_probe(AVProbeData *p)

{

    const unsigned char *buf = p->buf;

    unsigned c, length = 0;



    // stream must start with packet of type 0 and sequence number 0

    if (*buf++ != 0)

        return 0;



    // read at most 2 bytes of coded length

    c = *buf++;

    length = c & 0x7F;

    if (c & 0x80) {

        c = *buf++;

        length = (length << 7) | (c & 0x7F);

    }

    if (c & 0x80 || length > 1024 || length < 21)

        return 0;



    if (memcmp(buf, "\r\nVersion:Vivo/", 15))

        return 0;

    buf += 15;



    if (*buf < '0' && *buf > '2')

        return 0;



    return AVPROBE_SCORE_MAX;

}
