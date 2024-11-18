static int adts_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    ADTSContext *adts = s->priv_data;

    AVIOContext *pb = s->pb;

    uint8_t buf[ADTS_HEADER_SIZE];



    if (!pkt->size)

        return 0;

    if (adts->write_adts) {

        ff_adts_write_frame_header(adts, buf, pkt->size, adts->pce_size);

        avio_write(pb, buf, ADTS_HEADER_SIZE);

        if (adts->pce_size) {

            avio_write(pb, adts->pce_data, adts->pce_size);

            adts->pce_size = 0;

        }

    }

    avio_write(pb, pkt->data, pkt->size);

    avio_flush(pb);



    return 0;

}
