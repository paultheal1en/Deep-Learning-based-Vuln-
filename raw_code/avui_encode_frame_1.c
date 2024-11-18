static int avui_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                             const AVFrame *pic, int *got_packet)

{

    uint8_t *dst, *src = pic->data[0];

    int i, j, skip, ret, size, interlaced;



    interlaced = avctx->field_order > AV_FIELD_PROGRESSIVE;



    if (avctx->height == 486) {

        skip = 10;

    } else {

        skip = 16;

    }

    size = 2 * avctx->width * (avctx->height + skip) + 8 * interlaced;

    if ((ret = ff_alloc_packet2(avctx, pkt, size)) < 0)

        return ret;

    dst = pkt->data;

    if (!(avctx->extradata = av_mallocz(24 + FF_INPUT_BUFFER_PADDING_SIZE)))

        return AVERROR(ENOMEM);

    avctx->extradata_size = 24;

    memcpy(avctx->extradata, "\0\0\0\x18""APRGAPRG0001", 16);

    if (interlaced) {

        avctx->extradata[19] = 2;

    } else {

        avctx->extradata[19] = 1;

        dst += avctx->width * skip;

    }



    avctx->coded_frame->reference = 0;

    avctx->coded_frame->key_frame = 1;

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;



    for (i = 0; i <= interlaced; i++) {

        if (interlaced && avctx->height == 486) {

            src = pic->data[0] + (1 - i) * pic->linesize[0];

        } else {

            src = pic->data[0] + i * pic->linesize[0];

        }

        dst += avctx->width * skip + 4 * i;

        for (j = 0; j < avctx->height; j += interlaced + 1) {

            memcpy(dst, src, avctx->width * 2);

            src += (interlaced + 1) * pic->linesize[0];

            dst += avctx->width * 2;

        }

    }



    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;

    return 0;

}
