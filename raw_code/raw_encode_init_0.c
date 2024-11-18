static av_cold int raw_encode_init(AVCodecContext *avctx)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);



    avctx->coded_frame            = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;

    avctx->bits_per_coded_sample = av_get_bits_per_pixel(desc);

    if(!avctx->codec_tag)

        avctx->codec_tag = avcodec_pix_fmt_to_codec_tag(avctx->pix_fmt);

    return 0;

}
