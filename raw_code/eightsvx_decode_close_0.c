static av_cold int eightsvx_decode_close(AVCodecContext *avctx)

{

    EightSvxContext *esc = avctx->priv_data;



    av_freep(&esc->samples);

    esc->samples_size = 0;

    esc->samples_idx = 0;



    return 0;

}
