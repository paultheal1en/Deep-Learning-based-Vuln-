static av_cold int v4l2_decode_init(AVCodecContext *avctx)

{

    V4L2m2mContext *s = avctx->priv_data;

    V4L2Context *capture = &s->capture;

    V4L2Context *output = &s->output;

    int ret;



    /* if these dimensions are invalid (ie, 0 or too small) an event will be raised

     * by the v4l2 driver; this event will trigger a full pipeline reconfig and

     * the proper values will be retrieved from the kernel driver.

     */

    output->height = capture->height = avctx->coded_height;

    output->width = capture->width = avctx->coded_width;



    output->av_codec_id = avctx->codec_id;

    output->av_pix_fmt  = AV_PIX_FMT_NONE;



    capture->av_codec_id = AV_CODEC_ID_RAWVIDEO;

    capture->av_pix_fmt = avctx->pix_fmt;



    ret = ff_v4l2_m2m_codec_init(avctx);

    if (ret) {

        av_log(avctx, AV_LOG_ERROR, "can't configure decoder\n");

        return ret;

    }



    return v4l2_prepare_decoder(s);

}
