static int scale_vaapi_filter_frame(AVFilterLink *inlink, AVFrame *input_frame)

{

    AVFilterContext *avctx = inlink->dst;

    AVFilterLink *outlink = avctx->outputs[0];

    ScaleVAAPIContext *ctx = avctx->priv;

    AVFrame *output_frame = NULL;

    VASurfaceID input_surface, output_surface;

    VAProcPipelineParameterBuffer params;

    VABufferID params_id;

    VAStatus vas;

    int err;



    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",

           av_get_pix_fmt_name(input_frame->format),

           input_frame->width, input_frame->height, input_frame->pts);



    if (ctx->va_context == VA_INVALID_ID)

        return AVERROR(EINVAL);



    input_surface = (VASurfaceID)(uintptr_t)input_frame->data[3];

    av_log(ctx, AV_LOG_DEBUG, "Using surface %#x for scale input.\n",

           input_surface);



    output_frame = av_frame_alloc();

    if (!output_frame) {

        av_log(ctx, AV_LOG_ERROR, "Failed to allocate output frame.");

        err = AVERROR(ENOMEM);

        goto fail;

    }



    err = av_hwframe_get_buffer(ctx->output_frames_ref, output_frame, 0);

    if (err < 0) {

        av_log(ctx, AV_LOG_ERROR, "Failed to get surface for "

               "output: %d\n.", err);

    }



    output_surface = (VASurfaceID)(uintptr_t)output_frame->data[3];

    av_log(ctx, AV_LOG_DEBUG, "Using surface %#x for scale output.\n",

           output_surface);



    memset(&params, 0, sizeof(params));



    params.surface = input_surface;

    params.surface_region = 0;

    params.surface_color_standard =

        vaapi_proc_colour_standard(input_frame->colorspace);



    params.output_region = 0;

    params.output_background_color = 0xff000000;

    params.output_color_standard = params.surface_color_standard;



    params.pipeline_flags = 0;

    params.filter_flags = VA_FILTER_SCALING_HQ;



    vas = vaBeginPicture(ctx->hwctx->display,

                         ctx->va_context, output_surface);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to attach new picture: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    vas = vaCreateBuffer(ctx->hwctx->display, ctx->va_context,

                         VAProcPipelineParameterBufferType,

                         sizeof(params), 1, &params, &params_id);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to create parameter buffer: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail_after_begin;

    }

    av_log(ctx, AV_LOG_DEBUG, "Pipeline parameter buffer is %#x.\n",

           params_id);



    vas = vaRenderPicture(ctx->hwctx->display, ctx->va_context,

                          &params_id, 1);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to render parameter buffer: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail_after_begin;

    }



    vas = vaEndPicture(ctx->hwctx->display, ctx->va_context);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to start picture processing: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail_after_render;

    }



    if (ctx->hwctx->driver_quirks &

        AV_VAAPI_DRIVER_QUIRK_RENDER_PARAM_BUFFERS) {

        vas = vaDestroyBuffer(ctx->hwctx->display, params_id);

        if (vas != VA_STATUS_SUCCESS) {

            av_log(ctx, AV_LOG_ERROR, "Failed to free parameter buffer: "

                   "%d (%s).\n", vas, vaErrorStr(vas));

            // And ignore.

        }

    }



    av_frame_copy_props(output_frame, input_frame);

    av_frame_free(&input_frame);



    av_log(ctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",

           av_get_pix_fmt_name(output_frame->format),

           output_frame->width, output_frame->height, output_frame->pts);



    return ff_filter_frame(outlink, output_frame);



    // We want to make sure that if vaBeginPicture has been called, we also

    // call vaRenderPicture and vaEndPicture.  These calls may well fail or

    // do something else nasty, but once we're in this failure case there

    // isn't much else we can do.

fail_after_begin:

    vaRenderPicture(ctx->hwctx->display, ctx->va_context, &params_id, 1);

fail_after_render:

    vaEndPicture(ctx->hwctx->display, ctx->va_context);

fail:

    av_frame_free(&input_frame);

    av_frame_free(&output_frame);

    return err;

}
