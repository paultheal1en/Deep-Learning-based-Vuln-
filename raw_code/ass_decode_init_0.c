static av_cold int ass_decode_init(AVCodecContext *avctx)

{

    avctx->subtitle_header = av_malloc(avctx->extradata_size);

    if (!avctx->extradata)

        return AVERROR(ENOMEM);

    memcpy(avctx->subtitle_header, avctx->extradata, avctx->extradata_size);

    avctx->subtitle_header_size = avctx->extradata_size;

    return 0;

}
