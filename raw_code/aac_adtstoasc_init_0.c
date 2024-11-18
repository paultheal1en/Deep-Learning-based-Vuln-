static int aac_adtstoasc_init(AVBSFContext *ctx)

{

    av_freep(&ctx->par_out->extradata);

    ctx->par_out->extradata_size = 0;



    return 0;

}
