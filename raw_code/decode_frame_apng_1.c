static int decode_frame_apng(AVCodecContext *avctx,

                        void *data, int *got_frame,

                        AVPacket *avpkt)

{

    PNGDecContext *const s = avctx->priv_data;

    int ret;

    AVFrame *p;



    ff_thread_release_buffer(avctx, &s->last_picture);

    FFSWAP(ThreadFrame, s->picture, s->last_picture);

    p = s->picture.f;



    if (!(s->state & PNG_IHDR)) {

        int side_data_size = 0;

        uint8_t *side_data = NULL;

        if (avpkt)

            side_data = av_packet_get_side_data(avpkt, AV_PKT_DATA_NEW_EXTRADATA, &side_data_size);



        if (side_data_size) {

            av_freep(&s->extra_data);

            s->extra_data = av_mallocz(side_data_size + AV_INPUT_BUFFER_PADDING_SIZE);

            if (!s->extra_data)

                return AVERROR(ENOMEM);

            s->extra_data_size = side_data_size;

            memcpy(s->extra_data, side_data, s->extra_data_size);

        }



        if (!s->extra_data_size)

            return AVERROR_INVALIDDATA;



        /* only init fields, there is no zlib use in extradata */

        s->zstream.zalloc = ff_png_zalloc;

        s->zstream.zfree  = ff_png_zfree;



        bytestream2_init(&s->gb, s->extra_data, s->extra_data_size);

        if ((ret = decode_frame_common(avctx, s, p, avpkt)) < 0)

            goto end;

    }



    /* reset state for a new frame */

    if ((ret = inflateInit(&s->zstream)) != Z_OK) {

        av_log(avctx, AV_LOG_ERROR, "inflateInit returned error %d\n", ret);

        ret = AVERROR_EXTERNAL;

        goto end;

    }

    s->y = 0;

    s->state &= ~(PNG_IDAT | PNG_ALLIMAGE);

    bytestream2_init(&s->gb, avpkt->data, avpkt->size);

    if ((ret = decode_frame_common(avctx, s, p, avpkt)) < 0)

        goto end;



    if (!(s->state & PNG_ALLIMAGE))

        av_log(avctx, AV_LOG_WARNING, "Frame did not contain a complete image\n");

    if (!(s->state & (PNG_ALLIMAGE|PNG_IDAT))) {

        ret = AVERROR_INVALIDDATA;

        goto end;

    }

    if ((ret = av_frame_ref(data, s->picture.f)) < 0)

        goto end;



    *got_frame = 1;

    ret = bytestream2_tell(&s->gb);



end:

    inflateEnd(&s->zstream);

    return ret;

}
