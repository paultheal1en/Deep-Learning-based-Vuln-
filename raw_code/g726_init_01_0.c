static av_cold int g726_init(AVCodecContext * avctx)

{

    AVG726Context* c = (AVG726Context*)avctx->priv_data;

    unsigned int index= (avctx->bit_rate + avctx->sample_rate/2) / avctx->sample_rate - 2;



    if (

        (avctx->bit_rate != 16000 && avctx->bit_rate != 24000 &&

         avctx->bit_rate != 32000 && avctx->bit_rate != 40000)) {

        av_log(avctx, AV_LOG_ERROR, "G726: unsupported audio format\n");

        return -1;

    }

    if (avctx->sample_rate != 8000 && avctx->strict_std_compliance>FF_COMPLIANCE_INOFFICIAL) {

        av_log(avctx, AV_LOG_ERROR, "G726: unsupported audio format\n");

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

    g726_reset(&c->c, index);

    c->code_size = c->c.tbls->bits;

    c->bit_buffer = 0;

    c->bits_left = 0;



    avctx->coded_frame = avcodec_alloc_frame();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);

    avctx->coded_frame->key_frame = 1;



    return 0;

}
