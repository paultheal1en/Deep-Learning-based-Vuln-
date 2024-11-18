static inline void gen_op_arith_compute_ov(DisasContext *ctx, TCGv arg0,

                                           TCGv arg1, TCGv arg2, int sub)

{

    TCGv t0 = tcg_temp_new();



    tcg_gen_xor_tl(cpu_ov, arg0, arg1);

    tcg_gen_xor_tl(t0, arg1, arg2);

    if (sub) {

        tcg_gen_and_tl(cpu_ov, cpu_ov, t0);

    } else {

        tcg_gen_andc_tl(cpu_ov, cpu_ov, t0);

    }

    tcg_temp_free(t0);

    if (NARROW_MODE(ctx)) {

        tcg_gen_ext32s_tl(cpu_ov, cpu_ov);

    }

    tcg_gen_shri_tl(cpu_ov, cpu_ov, TARGET_LONG_BITS - 1);

    tcg_gen_or_tl(cpu_so, cpu_so, cpu_ov);

}
