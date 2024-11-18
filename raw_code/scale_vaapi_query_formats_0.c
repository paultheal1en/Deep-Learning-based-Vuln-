static int scale_vaapi_query_formats(AVFilterContext *avctx)

{

    enum AVPixelFormat pix_fmts[] = {

        AV_PIX_FMT_VAAPI, AV_PIX_FMT_NONE,

    };



    ff_formats_ref(ff_make_format_list(pix_fmts),

                   &avctx->inputs[0]->out_formats);

    ff_formats_ref(ff_make_format_list(pix_fmts),

                   &avctx->outputs[0]->in_formats);



    return 0;

}
