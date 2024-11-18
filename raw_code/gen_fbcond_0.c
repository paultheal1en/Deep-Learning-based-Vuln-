static always_inline void gen_fbcond (DisasContext *ctx,

                                      void* func,

                                      int ra, int32_t disp16)

{

    int l1, l2;

    TCGv tmp;



    l1 = gen_new_label();

    l2 = gen_new_label();

    if (ra != 31) {

        tmp = tcg_temp_new(TCG_TYPE_I64);

        tcg_gen_helper_1_1(func, tmp, cpu_fir[ra]);

    } else  {

        tmp = tcg_const_i64(0);

        tcg_gen_helper_1_1(func, tmp, tmp);

    }

    tcg_gen_brcondi_i64(TCG_COND_NE, tmp, 0, l1);

    tcg_gen_movi_i64(cpu_pc, ctx->pc);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_i64(cpu_pc, ctx->pc + (int64_t)(disp16 << 2));

    gen_set_label(l2);

}
