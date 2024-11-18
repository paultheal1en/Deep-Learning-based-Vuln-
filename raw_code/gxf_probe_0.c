static int gxf_probe(AVProbeData *p) {

    static const uint8_t startcode[] = {0, 0, 0, 0, 1, 0xbc}; // start with map packet

    static const uint8_t endcode[] = {0, 0, 0, 0, 0xe1, 0xe2};

    if (p->buf_size < 16)

        return 0;

    if (!memcmp(p->buf, startcode, sizeof(startcode)) &&

        !memcmp(&p->buf[16 - sizeof(endcode)], endcode, sizeof(endcode)))

        return AVPROBE_SCORE_MAX;

    return 0;

}
