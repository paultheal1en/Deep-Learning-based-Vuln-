static int xan_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    XanContext *s = avctx->priv_data;

    AVPaletteControl *palette_control = avctx->palctrl;

    int keyframe = 0;



    if (palette_control->palette_changed) {

        /* load the new palette and reset the palette control */

        xan_wc3_build_palette(s, palette_control->palette);

        /* If pal8 we clear flag when we copy palette */

        if (s->avctx->pix_fmt != PIX_FMT_PAL8)

            palette_control->palette_changed = 0;

        keyframe = 1;

    }



    if (avctx->get_buffer(avctx, &s->current_frame)) {

        av_log(s->avctx, AV_LOG_ERROR, "  Xan Video: get_buffer() failed\n");

        return -1;

    }

    s->current_frame.reference = 3;



    s->buf = buf;

    s->size = buf_size;



    if (avctx->codec->id == CODEC_ID_XAN_WC3)

        xan_wc3_decode_frame(s);

    else if (avctx->codec->id == CODEC_ID_XAN_WC4)

        xan_wc4_decode_frame(s);



    /* release the last frame if it is allocated */

    if (s->last_frame.data[0])

        avctx->release_buffer(avctx, &s->last_frame);



    /* shuffle frames */

    s->last_frame = s->current_frame;



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->current_frame;



    /* always report that the buffer was completely consumed */

    return buf_size;

}
