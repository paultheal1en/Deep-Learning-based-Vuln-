static void spr_write_ibatl_h (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_ibatl((sprn - SPR_IBAT4L) / 2);

    RET_STOP(ctx);

}
