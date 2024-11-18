uint8_t *av_packet_new_side_data(AVPacket *pkt, enum AVPacketSideDataType type,

                                 int size)

{

    int ret;

    uint8_t *data;



    if ((unsigned)size > INT_MAX - AV_INPUT_BUFFER_PADDING_SIZE)

        return NULL;

    data = av_malloc(size + AV_INPUT_BUFFER_PADDING_SIZE);

    if (!data)

        return NULL;



    ret = av_packet_add_side_data(pkt, type, data, size);

    if (ret < 0) {

        av_freep(&data);

        return NULL;

    }



    return data;

}
