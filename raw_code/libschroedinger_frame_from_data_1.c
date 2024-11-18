static SchroFrame *libschroedinger_frame_from_data(AVCodecContext *avctx,

                                                   const AVFrame *frame)

{

    SchroEncoderParams *p_schro_params = avctx->priv_data;

    SchroFrame *in_frame = ff_create_schro_frame(avctx,

                                                 p_schro_params->frame_format);



    if (in_frame) {

        /* Copy input data to SchroFrame buffers (they match the ones

         * referenced by the AVFrame stored in priv) */

        if (av_frame_copy(in_frame->priv, frame) < 0) {

            av_log(avctx, AV_LOG_ERROR, "Failed to copy input data\n");

            return NULL;

        }

    }



    return in_frame;

}
