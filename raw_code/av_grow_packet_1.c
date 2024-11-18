int av_grow_packet(AVPacket *pkt, int grow_by)

{

    int new_size;

    av_assert0((unsigned)pkt->size <= INT_MAX - AV_INPUT_BUFFER_PADDING_SIZE);

    if ((unsigned)grow_by >

        INT_MAX - (pkt->size + AV_INPUT_BUFFER_PADDING_SIZE))

        return -1;



    new_size = pkt->size + grow_by + AV_INPUT_BUFFER_PADDING_SIZE;

    if (pkt->buf) {

        size_t data_offset;

        uint8_t *old_data = pkt->data;

        if (pkt->data == NULL) {

            data_offset = 0;

            pkt->data = pkt->buf->data;

        } else {

            data_offset = pkt->data - pkt->buf->data;

            if (data_offset > INT_MAX - new_size)

                return -1;

        }



        if (new_size + data_offset > pkt->buf->size) {

            int ret = av_buffer_realloc(&pkt->buf, new_size + data_offset);

            if (ret < 0) {

                pkt->data = old_data;

                return ret;

            }

            pkt->data = pkt->buf->data + data_offset;

        }

    } else {

        pkt->buf = av_buffer_alloc(new_size);

        if (!pkt->buf)

            return AVERROR(ENOMEM);

        memcpy(pkt->buf->data, pkt->data, pkt->size);

        pkt->data = pkt->buf->data;

    }

    pkt->size += grow_by;

    memset(pkt->data + pkt->size, 0, AV_INPUT_BUFFER_PADDING_SIZE);



    return 0;

}
