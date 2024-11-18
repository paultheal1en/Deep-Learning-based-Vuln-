static void spr_write_601_ubatu (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_601_batu((sprn - SPR_IBAT0U) / 2);

    RET_STOP(ctx);

}
