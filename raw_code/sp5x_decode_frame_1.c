static int sp5x_decode_frame(AVCodecContext *avctx,

                              void *data, int *data_size,

                              AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    AVPacket avpkt_recoded;

    const int qscale = 5;

    const uint8_t *buf_ptr;

    uint8_t *recoded;

    int i = 0, j = 0;



    if (!avctx->width || !avctx->height)

        return -1;



    buf_ptr = buf;



    recoded = av_mallocz(buf_size + 1024);

    if (!recoded)

        return -1;



    /* SOI */

    recoded[j++] = 0xFF;

    recoded[j++] = 0xD8;



    memcpy(recoded+j, &sp5x_data_dqt[0], sizeof(sp5x_data_dqt));

    memcpy(recoded+j+5, &sp5x_quant_table[qscale * 2], 64);

    memcpy(recoded+j+70, &sp5x_quant_table[(qscale * 2) + 1], 64);

    j += sizeof(sp5x_data_dqt);



    memcpy(recoded+j, &sp5x_data_dht[0], sizeof(sp5x_data_dht));

    j += sizeof(sp5x_data_dht);



    memcpy(recoded+j, &sp5x_data_sof[0], sizeof(sp5x_data_sof));

    AV_WB16(recoded+j+5, avctx->coded_height);

    AV_WB16(recoded+j+7, avctx->coded_width);

    j += sizeof(sp5x_data_sof);



    memcpy(recoded+j, &sp5x_data_sos[0], sizeof(sp5x_data_sos));

    j += sizeof(sp5x_data_sos);



    if(avctx->codec_id==CODEC_ID_AMV)

        for (i = 2; i < buf_size-2 && j < buf_size+1024-2; i++)

            recoded[j++] = buf[i];

    else

    for (i = 14; i < buf_size && j < buf_size+1024-2; i++)

    {

        recoded[j++] = buf[i];

        if (buf[i] == 0xff)

            recoded[j++] = 0;

    }



    /* EOI */

    recoded[j++] = 0xFF;

    recoded[j++] = 0xD9;



    avctx->flags &= ~CODEC_FLAG_EMU_EDGE;

    av_init_packet(&avpkt_recoded);

    avpkt_recoded.data = recoded;

    avpkt_recoded.size = j;

    i = ff_mjpeg_decode_frame(avctx, data, data_size, &avpkt_recoded);



    av_free(recoded);



    return i;

}
