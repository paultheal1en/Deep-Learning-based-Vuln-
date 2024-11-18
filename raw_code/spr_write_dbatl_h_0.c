static void spr_write_dbatl_h (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_dbatl((sprn - SPR_DBAT4L) / 2);

    RET_STOP(ctx);

}
