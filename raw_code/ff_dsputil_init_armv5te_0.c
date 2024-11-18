av_cold void ff_dsputil_init_armv5te(DSPContext *c, AVCodecContext *avctx)

{

    if (avctx->bits_per_raw_sample <= 8 &&

        (avctx->idct_algo == FF_IDCT_AUTO ||

         avctx->idct_algo == FF_IDCT_SIMPLEARMV5TE)) {

        c->idct_put              = ff_simple_idct_put_armv5te;

        c->idct_add              = ff_simple_idct_add_armv5te;

        c->idct                  = ff_simple_idct_armv5te;

        c->idct_permutation_type = FF_NO_IDCT_PERM;

    }



    c->prefetch = ff_prefetch_arm;

}
