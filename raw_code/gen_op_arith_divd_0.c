static inline void gen_op_arith_divd(DisasContext *ctx, TCGv ret, TCGv arg1,

                                     TCGv arg2, int sign, int compute_ov)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();



    tcg_gen_brcondi_i64(TCG_COND_EQ, arg2, 0, l1);

    if (sign) {

        int l3 = gen_new_label();

        tcg_gen_brcondi_i64(TCG_COND_NE, arg2, -1, l3);

        tcg_gen_brcondi_i64(TCG_COND_EQ, arg1, INT64_MIN, l1);

        gen_set_label(l3);

        tcg_gen_div_i64(ret, arg1, arg2);

    } else {

        tcg_gen_divu_i64(ret, arg1, arg2);

    }

    if (compute_ov) {

        tcg_gen_movi_tl(cpu_ov, 0);

    }

    tcg_gen_br(l2);

    gen_set_label(l1);

    if (sign) {

        tcg_gen_sari_i64(ret, arg1, 63);

    } else {

        tcg_gen_movi_i64(ret, 0);

    }

    if (compute_ov) {

        tcg_gen_movi_tl(cpu_ov, 1);

        tcg_gen_movi_tl(cpu_so, 1);

    }

    gen_set_label(l2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, ret);

}
