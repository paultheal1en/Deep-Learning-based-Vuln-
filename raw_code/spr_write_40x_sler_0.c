static void spr_write_40x_sler (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_40x_sler();

    /* We must stop the translation as we may have changed

     * some regions endianness

     */

    RET_STOP(ctx);

}
