static int read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    ByteIOContext *pb = s->pb;

    PutBitContext pbo;

    uint16_t buf[8 * MAX_FRAME_SIZE + 2];

    int packet_size;

    int sync;

    uint16_t* src=buf;

    int i, j, ret;



    if(url_feof(pb))

        return AVERROR_EOF;



    sync = get_le16(pb); // sync word

    packet_size = get_le16(pb) / 8;

    assert(packet_size < 8 * MAX_FRAME_SIZE);



    ret = get_buffer(pb, (uint8_t*)buf, (8 * packet_size) * sizeof(uint16_t));

    if(ret<0)

        return ret;

    if(ret != 8 * packet_size * sizeof(uint16_t))

        return AVERROR(EIO);



    av_new_packet(pkt, packet_size);



    init_put_bits(&pbo, pkt->data, packet_size);

    for(j=0; j < packet_size; j++)

        for(i=0; i<8;i++)

            put_bits(&pbo,1, AV_RL16(src++) == BIT_1 ? 1 : 0);



    flush_put_bits(&pbo);



    pkt->duration=1;

    return 0;

}
