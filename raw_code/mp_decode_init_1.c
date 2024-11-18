static av_cold int mp_decode_init(AVCodecContext *avctx)

{

    MotionPixelsContext *mp = avctx->priv_data;

    int w4 = (avctx->width  + 3) & ~3;

    int h4 = (avctx->height + 3) & ~3;



    if(avctx->extradata_size < 2){

        av_log(avctx, AV_LOG_ERROR, "extradata too small\n");

        return AVERROR_INVALIDDATA;

    }



    motionpixels_tableinit();

    mp->avctx = avctx;

    ff_dsputil_init(&mp->dsp, avctx);

    mp->changes_map = av_mallocz(avctx->width * h4);

    mp->offset_bits_len = av_log2(avctx->width * avctx->height) + 1;

    mp->vpt = av_mallocz(avctx->height * sizeof(YuvPixel));

    mp->hpt = av_mallocz(h4 * w4 / 16 * sizeof(YuvPixel));



    avctx->pix_fmt = AV_PIX_FMT_RGB555;

    avcodec_get_frame_defaults(&mp->frame);

    return 0;

}