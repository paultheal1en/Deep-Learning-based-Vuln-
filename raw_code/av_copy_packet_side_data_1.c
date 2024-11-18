int av_copy_packet_side_data(AVPacket *pkt, AVPacket *src)

{

    if (src->side_data_elems) {

        int i;

        DUP_DATA(pkt->side_data, src->side_data,

                src->side_data_elems * sizeof(*src->side_data), 0, ALLOC_MALLOC);

        memset(pkt->side_data, 0,

                src->side_data_elems * sizeof(*src->side_data));

        for (i = 0; i < src->side_data_elems; i++) {

            DUP_DATA(pkt->side_data[i].data, src->side_data[i].data,

                    src->side_data[i].size, 1, ALLOC_MALLOC);

            pkt->side_data[i].size = src->side_data[i].size;

            pkt->side_data[i].type = src->side_data[i].type;

        }

    }

    return 0;



failed_alloc:

    av_destruct_packet(pkt);

    return AVERROR(ENOMEM);

}
