static int decode_frame(AVCodecContext *avctx,

                        void *data, int *data_size,

                        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    TM2Context * const l = avctx->priv_data;

    AVFrame * const p= (AVFrame*)&l->pic;

    int i, skip, t;

    uint8_t *swbuf;



    swbuf = av_malloc(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);

    if(!swbuf){

        av_log(avctx, AV_LOG_ERROR, "Cannot allocate temporary buffer\n");

        return -1;

    }

    p->reference = 1;

    p->buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE | FF_BUFFER_HINTS_REUSABLE;

    if(avctx->reget_buffer(avctx, p) < 0){

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        av_free(swbuf);

        return -1;

    }



    l->dsp.bswap_buf((uint32_t*)swbuf, (const uint32_t*)buf, buf_size >> 2);

    skip = tm2_read_header(l, swbuf);



    if(skip == -1){

        av_free(swbuf);

        return -1;

    }



    for(i = 0; i < TM2_NUM_STREAMS; i++){

        t = tm2_read_stream(l, swbuf + skip, tm2_stream_order[i]);

        if(t == -1){

            av_free(swbuf);

            return -1;

        }

        skip += t;

    }

    p->key_frame = tm2_decode_blocks(l, p);

    if(p->key_frame)

        p->pict_type = FF_I_TYPE;

    else

        p->pict_type = FF_P_TYPE;



    l->cur = !l->cur;

    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = l->pic;

    av_free(swbuf);



    return buf_size;

}
