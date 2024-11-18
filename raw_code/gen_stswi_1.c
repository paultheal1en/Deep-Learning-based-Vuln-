static void gen_stswi(DisasContext *ctx)

{

    TCGv t0;

    TCGv_i32 t1, t2;

    int nb = NB(ctx->opcode);

    gen_set_access_type(ctx, ACCESS_INT);

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    t0 = tcg_temp_new();

    gen_addr_register(ctx, t0);

    if (nb == 0)

        nb = 32;

    t1 = tcg_const_i32(nb);

    t2 = tcg_const_i32(rS(ctx->opcode));

    gen_helper_stsw(cpu_env, t0, t1, t2);

    tcg_temp_free(t0);

    tcg_temp_free_i32(t1);

    tcg_temp_free_i32(t2);

}
