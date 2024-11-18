static int hwmap_config_output(AVFilterLink *outlink)

{

    AVFilterContext *avctx = outlink->src;

    HWMapContext      *ctx = avctx->priv;

    AVFilterLink   *inlink = avctx->inputs[0];

    AVHWFramesContext *hwfc;

    AVBufferRef *device;

    const AVPixFmtDescriptor *desc;

    int err;



    av_log(avctx, AV_LOG_DEBUG, "Configure hwmap %s -> %s.\n",

           av_get_pix_fmt_name(inlink->format),

           av_get_pix_fmt_name(outlink->format));



    av_buffer_unref(&ctx->hwframes_ref);



    device = avctx->hw_device_ctx;



    if (inlink->hw_frames_ctx) {

        hwfc = (AVHWFramesContext*)inlink->hw_frames_ctx->data;



        if (ctx->derive_device_type) {

            enum AVHWDeviceType type;



            type = av_hwdevice_find_type_by_name(ctx->derive_device_type);

            if (type == AV_HWDEVICE_TYPE_NONE) {

                av_log(avctx, AV_LOG_ERROR, "Invalid device type.\n");

                goto fail;

            }



            err = av_hwdevice_ctx_create_derived(&device, type,

                                                 hwfc->device_ref, 0);

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to created derived "

                       "device context: %d.\n", err);

                goto fail;

            }

        }



        desc = av_pix_fmt_desc_get(outlink->format);

        if (!desc) {

            err = AVERROR(EINVAL);

            goto fail;

        }



        if (inlink->format == hwfc->format &&

            (desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {

            // Map between two hardware formats (including the case of

            // undoing an existing mapping).



            if (!device) {

                av_log(avctx, AV_LOG_ERROR, "A device reference is "

                       "required to map to a hardware format.\n");

                err = AVERROR(EINVAL);

                goto fail;

            }



            err = av_hwframe_ctx_create_derived(&ctx->hwframes_ref,

                                                outlink->format,

                                                device,

                                                inlink->hw_frames_ctx, 0);

            if (err < 0) {

                av_log(avctx, AV_LOG_ERROR, "Failed to create derived "

                       "frames context: %d.\n", err);

                goto fail;

            }



        } else if ((outlink->format == hwfc->format &&

                    inlink->format  == hwfc->sw_format) ||

                   inlink->format == hwfc->format) {

            // Map from a hardware format to a software format, or

            // undo an existing such mapping.



            ctx->hwframes_ref = av_buffer_ref(inlink->hw_frames_ctx);

            if (!ctx->hwframes_ref) {

                err = AVERROR(ENOMEM);

                goto fail;

            }



        } else {

            // Non-matching formats - not supported.



            av_log(avctx, AV_LOG_ERROR, "Unsupported formats for "

                   "hwmap: from %s (%s) to %s.\n",

                   av_get_pix_fmt_name(inlink->format),

                   av_get_pix_fmt_name(hwfc->format),

                   av_get_pix_fmt_name(outlink->format));

            err = AVERROR(EINVAL);

            goto fail;

        }

    } else if (avctx->hw_device_ctx) {

        // Map from a software format to a hardware format.  This

        // creates a new hwframe context like hwupload, but then

        // returns frames mapped from that to the previous link in

        // order to fill them without an additional copy.



        if (!device) {

            av_log(avctx, AV_LOG_ERROR, "A device reference is "

                   "required to create new frames with backwards "

                   "mapping.\n");

            err = AVERROR(EINVAL);

            goto fail;

        }



        ctx->map_backwards = 1;



        ctx->hwframes_ref = av_hwframe_ctx_alloc(device);

        if (!ctx->hwframes_ref) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        hwfc = (AVHWFramesContext*)ctx->hwframes_ref->data;



        hwfc->format    = outlink->format;

        hwfc->sw_format = inlink->format;

        hwfc->width     = inlink->w;

        hwfc->height    = inlink->h;



        err = av_hwframe_ctx_init(ctx->hwframes_ref);

        if (err < 0) {

            av_log(avctx, AV_LOG_ERROR, "Failed to create frame "

                   "context for backward mapping: %d.\n", err);

            goto fail;

        }



    } else {

        av_log(avctx, AV_LOG_ERROR, "Mapping requires a hardware "

               "context (a device, or frames on input).\n");

        return AVERROR(EINVAL);

    }



    outlink->hw_frames_ctx = av_buffer_ref(ctx->hwframes_ref);

    if (!outlink->hw_frames_ctx) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    outlink->w = inlink->w;

    outlink->h = inlink->h;



    return 0;



fail:

    av_buffer_unref(&ctx->hwframes_ref);

    return err;

}
