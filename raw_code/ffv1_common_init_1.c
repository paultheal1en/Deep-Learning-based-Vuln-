av_cold int ffv1_common_init(AVCodecContext *avctx)

{

    FFV1Context *s = avctx->priv_data;



    if (!avctx->width || !avctx->height)

        return AVERROR_INVALIDDATA;



    s->avctx = avctx;

    s->flags = avctx->flags;



    s->picture.f = avcodec_alloc_frame();

    s->last_picture.f = av_frame_alloc();



    ff_dsputil_init(&s->dsp, avctx);



    s->width  = avctx->width;

    s->height = avctx->height;



    // defaults

    s->num_h_slices = 1;

    s->num_v_slices = 1;



    return 0;

}