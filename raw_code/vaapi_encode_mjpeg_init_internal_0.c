static av_cold int vaapi_encode_mjpeg_init_internal(AVCodecContext *avctx)

{

    static const VAConfigAttrib default_config_attributes[] = {

        { .type  = VAConfigAttribRTFormat,

          .value = VA_RT_FORMAT_YUV420 },

        { .type  = VAConfigAttribEncPackedHeaders,

          .value = VA_ENC_PACKED_HEADER_SEQUENCE },

    };



    VAAPIEncodeContext       *ctx = avctx->priv_data;

    VAAPIEncodeMJPEGContext *priv = ctx->priv_data;

    int i;



    ctx->va_profile    = VAProfileJPEGBaseline;

    ctx->va_entrypoint = VAEntrypointEncPicture;



    ctx->input_width    = avctx->width;

    ctx->input_height   = avctx->height;

    ctx->aligned_width  = FFALIGN(ctx->input_width,  8);

    ctx->aligned_height = FFALIGN(ctx->input_height, 8);



    for (i = 0; i < FF_ARRAY_ELEMS(default_config_attributes); i++) {

        ctx->config_attributes[ctx->nb_config_attributes++] =

            default_config_attributes[i];

    }



    priv->quality = avctx->global_quality;

    if (priv->quality < 1 || priv->quality > 100) {

        av_log(avctx, AV_LOG_ERROR, "Invalid quality value %d "

               "(must be 1-100).\n", priv->quality);

        return AVERROR(EINVAL);

    }



    vaapi_encode_mjpeg_init_tables(avctx);



    return 0;

}
