static always_inline void gen_bcond (DisasContext *ctx,

                                     TCGCond cond,

                                     int ra, int32_t disp16, int mask)

{

    int l1, l2;



    l1 = gen_new_label();

    l2 = gen_new_label();

    if (likely(ra != 31)) {

        if (mask) {

            TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

            tcg_gen_andi_i64(tmp, cpu_ir[ra], 1);

            tcg_gen_brcondi_i64(cond, tmp, 0, l1);

            tcg_temp_free(tmp);

        } else

            tcg_gen_brcondi_i64(cond, cpu_ir[ra], 0, l1);

    } else {

        /* Very uncommon case - Do not bother to optimize.  */

        TCGv tmp = tcg_const_i64(0);

        tcg_gen_brcondi_i64(cond, tmp, 0, l1);

        tcg_temp_free(tmp);

    }

    tcg_gen_movi_i64(cpu_pc, ctx->pc);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_i64(cpu_pc, ctx->pc + (int64_t)(disp16 << 2));

    gen_set_label(l2);

}
