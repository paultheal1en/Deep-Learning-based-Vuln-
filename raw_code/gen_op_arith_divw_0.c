static inline void gen_op_arith_divw(DisasContext *ctx, TCGv ret, TCGv arg1,

                                     TCGv arg2, int sign, int compute_ov)

{

    TCGLabel *l1 = gen_new_label();

    TCGLabel *l2 = gen_new_label();

    TCGv_i32 t0 = tcg_temp_local_new_i32();

    TCGv_i32 t1 = tcg_temp_local_new_i32();



    tcg_gen_trunc_tl_i32(t0, arg1);

    tcg_gen_trunc_tl_i32(t1, arg2);

    tcg_gen_brcondi_i32(TCG_COND_EQ, t1, 0, l1);

    if (sign) {

        TCGLabel *l3 = gen_new_label();

        tcg_gen_brcondi_i32(TCG_COND_NE, t1, -1, l3);

        tcg_gen_brcondi_i32(TCG_COND_EQ, t0, INT32_MIN, l1);

        gen_set_label(l3);

        tcg_gen_div_i32(t0, t0, t1);

    } else {

        tcg_gen_divu_i32(t0, t0, t1);

    }

    if (compute_ov) {

        tcg_gen_movi_tl(cpu_ov, 0);

    }

    tcg_gen_br(l2);

    gen_set_label(l1);

    if (sign) {

        tcg_gen_sari_i32(t0, t0, 31);

    } else {

        tcg_gen_movi_i32(t0, 0);

    }

    if (compute_ov) {

        tcg_gen_movi_tl(cpu_ov, 1);

        tcg_gen_movi_tl(cpu_so, 1);

    }

    gen_set_label(l2);

    tcg_gen_extu_i32_tl(ret, t0);

    tcg_temp_free_i32(t0);

    tcg_temp_free_i32(t1);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, ret);

}
