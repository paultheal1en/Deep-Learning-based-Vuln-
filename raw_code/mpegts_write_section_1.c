static void mpegts_write_section(MpegTSSection *s, uint8_t *buf, int len)

{

    unsigned int crc;

    unsigned char packet[TS_PACKET_SIZE];

    const unsigned char *buf_ptr;

    unsigned char *q;

    int first, b, len1, left;



    crc = av_bswap32(av_crc(av_crc_get_table(AV_CRC_32_IEEE),

                            -1, buf, len - 4));



    buf[len - 4] = (crc >> 24) & 0xff;

    buf[len - 3] = (crc >> 16) & 0xff;

    buf[len - 2] = (crc >>  8) & 0xff;

    buf[len - 1] =  crc        & 0xff;



    /* send each packet */

    buf_ptr = buf;

    while (len > 0) {

        first = buf == buf_ptr;

        q     = packet;

        *q++  = 0x47;

        b     = s->pid >> 8;

        if (first)

            b |= 0x40;

        *q++  = b;

        *q++  = s->pid;

        s->cc = s->cc + 1 & 0xf;

        *q++  = 0x10 | s->cc;







        if (first)

            *q++ = 0; /* 0 offset */

        len1 = TS_PACKET_SIZE - (q - packet);

        if (len1 > len)

            len1 = len;

        memcpy(q, buf_ptr, len1);

        q += len1;

        /* add known padding data */

        left = TS_PACKET_SIZE - (q - packet);

        if (left > 0)

            memset(q, 0xff, left);



        s->write_packet(s, packet);



        buf_ptr += len1;

        len     -= len1;

