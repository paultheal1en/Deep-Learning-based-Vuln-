static av_cold int alac_decode_init(AVCodecContext * avctx)

{

    ALACContext *alac = avctx->priv_data;

    alac->avctx = avctx;

    alac->context_initialized = 0;



    alac->numchannels = alac->avctx->channels;



    return 0;

}
