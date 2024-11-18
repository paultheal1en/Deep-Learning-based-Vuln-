static av_cold int libschroedinger_decode_init(AVCodecContext *avctx)

{



    SchroDecoderParams *p_schro_params = avctx->priv_data;

    /* First of all, initialize our supporting libraries. */

    schro_init();



    schro_debug_set_level(avctx->debug);

    p_schro_params->decoder = schro_decoder_new();

    schro_decoder_set_skip_ratio(p_schro_params->decoder, 1);



    if (!p_schro_params->decoder)

        return -1;



    /* Initialize the decoded frame queue. */

    ff_schro_queue_init(&p_schro_params->dec_frame_queue);

    return 0;

}
