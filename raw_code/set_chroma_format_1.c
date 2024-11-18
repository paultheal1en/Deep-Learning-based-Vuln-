static int set_chroma_format(AVCodecContext *avctx)

{

    int num_formats = sizeof(schro_pixel_format_map) /

                      sizeof(schro_pixel_format_map[0]);

    int idx;



    SchroEncoderParams *p_schro_params = avctx->priv_data;



    for (idx = 0; idx < num_formats; ++idx) {

        if (schro_pixel_format_map[idx].ff_pix_fmt == avctx->pix_fmt) {

            p_schro_params->format->chroma_format =

                            schro_pixel_format_map[idx].schro_pix_fmt;

            return 0;

        }

    }



    av_log(avctx, AV_LOG_ERROR,

           "This codec currently only supports planar YUV 4:2:0, 4:2:2"

           " and 4:4:4 formats.\n");



    return -1;

}
