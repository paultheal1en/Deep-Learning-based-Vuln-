static int read_packet(ByteIOContext *pb, uint8_t *buf, int raw_packet_size)

{

    int skip, len;



    for(;;) {

        len = get_buffer(pb, buf, TS_PACKET_SIZE);

        if (len != TS_PACKET_SIZE)

            return AVERROR(EIO);

        /* check paquet sync byte */

        if (buf[0] != 0x47) {

            /* find a new packet start */

            url_fseek(pb, -TS_PACKET_SIZE, SEEK_CUR);

            if (mpegts_resync(pb) < 0)

                return AVERROR_INVALIDDATA;

            else

                continue;

        } else {

            skip = raw_packet_size - TS_PACKET_SIZE;

            if (skip > 0)

                url_fskip(pb, skip);

            break;

        }

    }

    return 0;

}
