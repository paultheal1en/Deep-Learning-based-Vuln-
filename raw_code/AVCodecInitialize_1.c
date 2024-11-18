static AVCodec *AVCodecInitialize(enum AVCodecID codec_id)

{

    AVCodec *res;

    avcodec_register_all();

    av_log_set_level(AV_LOG_PANIC);

    res = avcodec_find_decoder(codec_id);

    if (!res)

        error("Failed to find decoder");

    return res;

}
