static av_cold int libschroedinger_decode_close(AVCodecContext *avctx)

{

    SchroDecoderParams *p_schro_params = avctx->priv_data;

    /* Free the decoder. */

    schro_decoder_free(p_schro_params->decoder);

    av_freep(&p_schro_params->format);



    /* Free data in the output frame queue. */

    ff_schro_queue_free(&p_schro_params->dec_frame_queue,

                        libschroedinger_decode_frame_free);



    return 0;

}
