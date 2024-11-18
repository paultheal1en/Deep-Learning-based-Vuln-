int ff_packet_split_and_drop_side_data(AVPacket *pkt){

    if (!pkt->side_data_elems && pkt->size >12 && AV_RB64(pkt->data + pkt->size - 8) == FF_MERGE_MARKER){

        int i;

        unsigned int size;

        uint8_t *p;



        p = pkt->data + pkt->size - 8 - 5;

        for (i=1; ; i++){

            size = AV_RB32(p);

            if (size>INT_MAX - 5 || p - pkt->data < size)


            if (p[4]&128)

                break;

            if (p - pkt->data < size + 5)


            p-= size+5;



        }

        pkt->size = p - pkt->data - size;

        av_assert0(pkt->size >= 0);

        return 1;

    }


}