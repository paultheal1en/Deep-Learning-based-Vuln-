static av_cold int g726_init(AVCodecContext * avctx)

{

    G726Context* c = avctx->priv_data;

    unsigned int index= (avctx->bit_rate + avctx->sample_rate/2) / avctx->sample_rate - 2;



    if (avctx->bit_rate % avctx->sample_rate && avctx->codec->encode) {

        av_log(avctx, AV_LOG_ERROR, "Bitrate - Samplerate combination is invalid\n");

        return -1;

    }

    if(avctx->channels != 1){

        av_log(avctx, AV_LOG_ERROR, "Only mono is supported\n");

        return -1;

    }

    if(index>3){

        av_log(avctx, AV_LOG_ERROR, "Unsupported number of bits %d\n", index+2);

        return -1;

    }

    g726_reset(c, index);

    c->code_size = index+2;



    avctx->coded_frame = avcodec_alloc_frame();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);

    avctx->coded_frame->key_frame = 1;



    if (avctx->codec->decode)

        avctx->sample_fmt = SAMPLE_FMT_S16;



    return 0;

}
