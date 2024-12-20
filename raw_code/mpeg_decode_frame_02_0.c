static int mpeg_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    Mpeg1Context *s = avctx->priv_data;

    AVFrame *picture = data;

    MpegEncContext *s2 = &s->mpeg_enc_ctx;

    av_dlog(avctx, "fill_buffer\n");



    if (buf_size == 0 || (buf_size == 4 && AV_RB32(buf) == SEQ_END_CODE)) {

        /* special case for last picture */

        if (s2->low_delay == 0 && s2->next_picture_ptr) {

            *picture = s2->next_picture_ptr->f;

            s2->next_picture_ptr = NULL;



            *data_size = sizeof(AVFrame);

        }

        return buf_size;

    }



    if (s2->flags & CODEC_FLAG_TRUNCATED) {

        int next = ff_mpeg1_find_frame_end(&s2->parse_context, buf, buf_size, NULL);



        if (ff_combine_frame(&s2->parse_context, next, (const uint8_t **)&buf, &buf_size) < 0)

            return buf_size;

    }



    s2->codec_tag = avpriv_toupper4(avctx->codec_tag);

    if (s->mpeg_enc_ctx_allocated == 0 && (   s2->codec_tag == AV_RL32("VCR2")

                                           || s2->codec_tag == AV_RL32("BW10")

                                          ))

        vcr2_init_sequence(avctx);



    s->slice_count = 0;



    if (avctx->extradata && !avctx->frame_number) {

        int ret = decode_chunks(avctx, picture, data_size, avctx->extradata, avctx->extradata_size);

        if(*data_size) {

            av_log(avctx, AV_LOG_ERROR, "picture in extradata\n");

            *data_size = 0;

        }

        if (ret < 0 && (avctx->err_recognition & AV_EF_EXPLODE))

            return ret;

    }



    return decode_chunks(avctx, picture, data_size, buf, buf_size);

}
