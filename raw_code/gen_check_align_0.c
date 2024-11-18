static inline void gen_check_align(DisasContext *ctx, TCGv EA, int mask)

{

    int l1 = gen_new_label();

    TCGv t0 = tcg_temp_new();

    TCGv_i32 t1, t2;

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    tcg_gen_andi_tl(t0, EA, mask);

    tcg_gen_brcondi_tl(TCG_COND_EQ, t0, 0, l1);

    t1 = tcg_const_i32(POWERPC_EXCP_ALIGN);

    t2 = tcg_const_i32(0);

    gen_helper_raise_exception_err(cpu_env, t1, t2);

    tcg_temp_free_i32(t1);

    tcg_temp_free_i32(t2);

    gen_set_label(l1);

    tcg_temp_free(t0);

}
