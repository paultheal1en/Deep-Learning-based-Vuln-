static always_inline void gen_excp (DisasContext *ctx,

                                    int exception, int error_code)

{

    TCGv tmp1, tmp2;



    tcg_gen_movi_i64(cpu_pc, ctx->pc);

    tmp1 = tcg_const_i32(exception);

    tmp2 = tcg_const_i32(error_code);

    tcg_gen_helper_0_2(helper_excp, tmp1, tmp2);

    tcg_temp_free(tmp2);

    tcg_temp_free(tmp1);

}
