static av_cold int libwebp_anim_encode_init(AVCodecContext *avctx)

{

    int ret = ff_libwebp_encode_init_common(avctx);

    if (!ret) {

        LibWebPAnimContext *s = avctx->priv_data;

        WebPAnimEncoderOptions enc_options;

        WebPAnimEncoderOptionsInit(&enc_options);

        // TODO(urvang): Expose some options on command-line perhaps.

        s->enc = WebPAnimEncoderNew(avctx->width, avctx->height, &enc_options);

        if (!s->enc)

            return AVERROR(EINVAL);

        s->prev_frame_pts = -1;

        s->done = 0;

    }

    return ret;

}
