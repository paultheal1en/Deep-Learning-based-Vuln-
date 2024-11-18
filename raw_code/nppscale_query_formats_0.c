static int nppscale_query_formats(AVFilterContext *ctx)

{

    static const enum AVPixelFormat pixel_formats[] = {

        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,

    };

    AVFilterFormats *pix_fmts  = ff_make_format_list(pixel_formats);



    ff_set_common_formats(ctx, pix_fmts);



    return 0;

}
