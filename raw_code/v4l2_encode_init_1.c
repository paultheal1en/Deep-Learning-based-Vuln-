static av_cold int v4l2_encode_init(AVCodecContext *avctx)

{

    V4L2m2mContext *s = avctx->priv_data;

    V4L2Context *capture = &s->capture;

    V4L2Context *output = &s->output;

    int ret;



    /* common settings output/capture */

    output->height = capture->height = avctx->height;

    output->width = capture->width = avctx->width;



    /* output context */

    output->av_codec_id = AV_CODEC_ID_RAWVIDEO;

    output->av_pix_fmt = avctx->pix_fmt;



    /* capture context */

    capture->av_codec_id = avctx->codec_id;

    capture->av_pix_fmt = AV_PIX_FMT_NONE;



    ret = ff_v4l2_m2m_codec_init(avctx);

    if (ret) {

        av_log(avctx, AV_LOG_ERROR, "can't configure encoder\n");

        return ret;

    }



    return v4l2_prepare_encoder(s);

}
