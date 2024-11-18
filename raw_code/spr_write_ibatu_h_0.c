static void spr_write_ibatu_h (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_ibatu((sprn - SPR_IBAT4U) / 2);

    RET_STOP(ctx);

}
