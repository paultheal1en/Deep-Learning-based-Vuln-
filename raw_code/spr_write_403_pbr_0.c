static void spr_write_403_pbr (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_403_pb(sprn - SPR_403_PBL1);

    RET_STOP(ctx);

}
