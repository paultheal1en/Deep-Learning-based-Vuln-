static av_cold int dnxhd_init_rc(DNXHDEncContext *ctx)

{

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->mb_rc, 8160*ctx->m.avctx->qmax*sizeof(RCEntry), fail);

    if (ctx->m.avctx->mb_decision != FF_MB_DECISION_RD)

        FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->mb_cmp, ctx->m.mb_num*sizeof(RCCMPEntry), fail);



    ctx->frame_bits = (ctx->cid_table->coding_unit_size - 640 - 4 - ctx->min_padding) * 8;

    ctx->qscale = 1;

    ctx->lambda = 2<<LAMBDA_FRAC_BITS; // qscale 2

    return 0;

 fail:

    return -1;

}
