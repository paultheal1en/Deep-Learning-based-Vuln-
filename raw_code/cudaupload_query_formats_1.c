static int cudaupload_query_formats(AVFilterContext *ctx)

{

    int ret;



    static const enum AVPixelFormat input_pix_fmts[] = {

        AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV444P,

        AV_PIX_FMT_NONE,

    };

    static const enum AVPixelFormat output_pix_fmts[] = {

        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,

    };

    AVFilterFormats *in_fmts  = ff_make_format_list(input_pix_fmts);

    AVFilterFormats *out_fmts = ff_make_format_list(output_pix_fmts);



    ret = ff_formats_ref(in_fmts, &ctx->inputs[0]->out_formats);

    if (ret < 0)

        return ret;



    ret = ff_formats_ref(out_fmts, &ctx->outputs[0]->in_formats);

    if (ret < 0)

        return ret;



    return 0;

}
