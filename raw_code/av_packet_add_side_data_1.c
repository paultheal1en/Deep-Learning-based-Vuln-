int av_packet_add_side_data(AVPacket *pkt, enum AVPacketSideDataType type,

                            uint8_t *data, size_t size)

{

    int elems = pkt->side_data_elems;



    if ((unsigned)elems + 1 > INT_MAX / sizeof(*pkt->side_data))

        return AVERROR(ERANGE);



    pkt->side_data = av_realloc(pkt->side_data,

                                (elems + 1) * sizeof(*pkt->side_data));

    if (!pkt->side_data)

        return AVERROR(ENOMEM);



    pkt->side_data[elems].data = data;

    pkt->side_data[elems].size = size;

    pkt->side_data[elems].type = type;

    pkt->side_data_elems++;



    return 0;

}
