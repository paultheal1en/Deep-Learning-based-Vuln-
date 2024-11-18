static void gen_movci (DisasContext *ctx, int rd, int rs, int cc, int tf)

{

    int l1 = gen_new_label();

    uint32_t ccbit;

    TCGCond cond;

    TCGv t0 = tcg_temp_local_new(TCG_TYPE_TL);

    TCGv t1 = tcg_temp_local_new(TCG_TYPE_TL);

    TCGv r_tmp = tcg_temp_local_new(TCG_TYPE_I32);



    if (cc)

        ccbit = 1 << (24 + cc);

    else

        ccbit = 1 << 23;

    if (tf)

        cond = TCG_COND_EQ;

    else

        cond = TCG_COND_NE;



    gen_load_gpr(t0, rd);

    gen_load_gpr(t1, rs);

    tcg_gen_andi_i32(r_tmp, fpu_fcr31, ccbit);

    tcg_gen_brcondi_i32(cond, r_tmp, 0, l1);

    tcg_temp_free(r_tmp);



    tcg_gen_mov_tl(t0, t1);

    tcg_temp_free(t1);



    gen_set_label(l1);

    gen_store_gpr(t0, rd);

    tcg_temp_free(t0);

}
