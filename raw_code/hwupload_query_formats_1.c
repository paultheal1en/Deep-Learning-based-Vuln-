static int hwupload_query_formats(AVFilterContext *avctx)

{

    HWUploadContext *ctx = avctx->priv;

    AVHWFramesConstraints *constraints = NULL;

    const enum AVPixelFormat *input_pix_fmts, *output_pix_fmts;

    AVFilterFormats *input_formats = NULL;

    int err, i;



    if (!avctx->hw_device_ctx) {

        av_log(ctx, AV_LOG_ERROR, "A hardware device reference is required "

               "to upload frames to.\n");

        return AVERROR(EINVAL);

    }



    ctx->hwdevice_ref = av_buffer_ref(avctx->hw_device_ctx);

    if (!ctx->hwdevice_ref)

        return AVERROR(ENOMEM);

    ctx->hwdevice = (AVHWDeviceContext*)ctx->hwdevice_ref->data;



    constraints = av_hwdevice_get_hwframe_constraints(ctx->hwdevice_ref, NULL);

    if (!constraints) {

        err = AVERROR(EINVAL);

        goto fail;

    }



    input_pix_fmts  = constraints->valid_sw_formats;

    output_pix_fmts = constraints->valid_hw_formats;



    input_formats = ff_make_format_list(output_pix_fmts);

    if (!input_formats) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    if (input_pix_fmts) {

        for (i = 0; input_pix_fmts[i] != AV_PIX_FMT_NONE; i++) {

            err = ff_add_format(&input_formats, input_pix_fmts[i]);

            if (err < 0) {

                ff_formats_unref(&input_formats);

                goto fail;

            }

        }

    }



    ff_formats_ref(input_formats, &avctx->inputs[0]->out_formats);



    ff_formats_ref(ff_make_format_list(output_pix_fmts),

                   &avctx->outputs[0]->in_formats);



    av_hwframe_constraints_free(&constraints);

    return 0;



fail:

    av_buffer_unref(&ctx->hwdevice_ref);

    av_hwframe_constraints_free(&constraints);

    return err;

}
