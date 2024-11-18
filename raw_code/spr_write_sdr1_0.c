static void spr_write_sdr1 (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_sdr1();

    RET_STOP(ctx);

}
