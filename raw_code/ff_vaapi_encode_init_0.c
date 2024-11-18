av_cold int ff_vaapi_encode_init(AVCodecContext *avctx,

                                 const VAAPIEncodeType *type)

{

    VAAPIEncodeContext *ctx = avctx->priv_data;

    AVVAAPIFramesContext *recon_hwctx = NULL;

    AVVAAPIHWConfig *hwconfig = NULL;

    AVHWFramesConstraints *constraints = NULL;

    enum AVPixelFormat recon_format;

    VAStatus vas;

    int err, i;



    if (!avctx->hw_frames_ctx) {

        av_log(avctx, AV_LOG_ERROR, "A hardware frames reference is "

               "required to associate the encoding device.\n");

        return AVERROR(EINVAL);

    }



    ctx->codec = type;

    ctx->codec_options = ctx->codec_options_data;



    ctx->va_config  = VA_INVALID_ID;

    ctx->va_context = VA_INVALID_ID;



    ctx->priv_data = av_mallocz(type->priv_data_size);

    if (!ctx->priv_data) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    ctx->input_frames_ref = av_buffer_ref(avctx->hw_frames_ctx);

    if (!ctx->input_frames_ref) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    ctx->input_frames = (AVHWFramesContext*)ctx->input_frames_ref->data;



    ctx->device_ref = av_buffer_ref(ctx->input_frames->device_ref);

    if (!ctx->device_ref) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    ctx->device = (AVHWDeviceContext*)ctx->device_ref->data;

    ctx->hwctx = ctx->device->hwctx;



    err = ctx->codec->init(avctx);

    if (err < 0)

        goto fail;



    err = vaapi_encode_check_config(avctx);

    if (err < 0)

        goto fail;



    vas = vaCreateConfig(ctx->hwctx->display,

                         ctx->va_profile, ctx->va_entrypoint,

                         ctx->config_attributes, ctx->nb_config_attributes,

                         &ctx->va_config);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to create encode pipeline "

               "configuration: %d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    hwconfig = av_hwdevice_hwconfig_alloc(ctx->device_ref);

    if (!hwconfig) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    hwconfig->config_id = ctx->va_config;



    constraints = av_hwdevice_get_hwframe_constraints(ctx->device_ref,

                                                      hwconfig);

    if (!constraints) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    // Probably we can use the input surface format as the surface format

    // of the reconstructed frames.  If not, we just pick the first (only?)

    // format in the valid list and hope that it all works.

    recon_format = AV_PIX_FMT_NONE;

    if (constraints->valid_sw_formats) {

        for (i = 0; constraints->valid_sw_formats[i] != AV_PIX_FMT_NONE; i++) {

            if (ctx->input_frames->sw_format ==

                constraints->valid_sw_formats[i]) {

                recon_format = ctx->input_frames->sw_format;

                break;

            }

        }

        if (recon_format == AV_PIX_FMT_NONE) {

            // No match.  Just use the first in the supported list and

            // hope for the best.

            recon_format = constraints->valid_sw_formats[0];

        }

    } else {

        // No idea what to use; copy input format.

        recon_format = ctx->input_frames->sw_format;

    }

    av_log(avctx, AV_LOG_DEBUG, "Using %s as format of "

           "reconstructed frames.\n", av_get_pix_fmt_name(recon_format));



    if (ctx->aligned_width  < constraints->min_width  ||

        ctx->aligned_height < constraints->min_height ||

        ctx->aligned_width  > constraints->max_width ||

        ctx->aligned_height > constraints->max_height) {

        av_log(avctx, AV_LOG_ERROR, "Hardware does not support encoding at "

               "size %dx%d (constraints: width %d-%d height %d-%d).\n",

               ctx->aligned_width, ctx->aligned_height,

               constraints->min_width,  constraints->max_width,

               constraints->min_height, constraints->max_height);

        err = AVERROR(EINVAL);

        goto fail;

    }



    av_freep(&hwconfig);

    av_hwframe_constraints_free(&constraints);



    ctx->recon_frames_ref = av_hwframe_ctx_alloc(ctx->device_ref);

    if (!ctx->recon_frames_ref) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    ctx->recon_frames = (AVHWFramesContext*)ctx->recon_frames_ref->data;



    ctx->recon_frames->format    = AV_PIX_FMT_VAAPI;

    ctx->recon_frames->sw_format = recon_format;

    ctx->recon_frames->width     = ctx->aligned_width;

    ctx->recon_frames->height    = ctx->aligned_height;

    ctx->recon_frames->initial_pool_size = ctx->nb_recon_frames;



    err = av_hwframe_ctx_init(ctx->recon_frames_ref);

    if (err < 0) {

        av_log(avctx, AV_LOG_ERROR, "Failed to initialise reconstructed "

               "frame context: %d.\n", err);

        goto fail;

    }

    recon_hwctx = ctx->recon_frames->hwctx;



    vas = vaCreateContext(ctx->hwctx->display, ctx->va_config,

                          ctx->aligned_width, ctx->aligned_height,

                          VA_PROGRESSIVE,

                          recon_hwctx->surface_ids,

                          recon_hwctx->nb_surfaces,

                          &ctx->va_context);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to create encode pipeline "

               "context: %d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    ctx->input_order  = 0;

    ctx->output_delay = avctx->max_b_frames;

    ctx->decode_delay = 1;

    ctx->output_order = - ctx->output_delay - 1;



    if (ctx->codec->sequence_params_size > 0) {

        ctx->codec_sequence_params =

            av_mallocz(ctx->codec->sequence_params_size);

        if (!ctx->codec_sequence_params) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

    }

    if (ctx->codec->picture_params_size > 0) {

        ctx->codec_picture_params =

            av_mallocz(ctx->codec->picture_params_size);

        if (!ctx->codec_picture_params) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

    }



    if (ctx->codec->init_sequence_params) {

        err = ctx->codec->init_sequence_params(avctx);

        if (err < 0) {

            av_log(avctx, AV_LOG_ERROR, "Codec sequence initialisation "

                   "failed: %d.\n", err);

            goto fail;

        }

    }



    ctx->output_buffer_pool =

        av_buffer_pool_init2(sizeof(VABufferID), avctx,

                             &vaapi_encode_alloc_output_buffer, NULL);

    if (!ctx->output_buffer_pool) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    // All I are IDR for now.

    ctx->i_per_idr = 0;

    ctx->p_per_i = ((avctx->gop_size + avctx->max_b_frames) /

                    (avctx->max_b_frames + 1));

    ctx->b_per_p = avctx->max_b_frames;



    // This should be configurable somehow.  (Needs testing on a machine

    // where it actually overlaps properly, though.)

    ctx->issue_mode = ISSUE_MODE_MAXIMISE_THROUGHPUT;



    return 0;



fail:

    av_freep(&hwconfig);

    av_hwframe_constraints_free(&constraints);

    ff_vaapi_encode_close(avctx);

    return err;

}
