static inline void gen_evmwumia(DisasContext *ctx)

{

    TCGv_i64 tmp;



    if (unlikely(!ctx->spe_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_APU);

        return;

    }



    gen_evmwumi(ctx);            /* rD := rA * rB */



    tmp = tcg_temp_new_i64();



    /* acc := rD */

    gen_load_gpr64(tmp, rD(ctx->opcode));

    tcg_gen_st_i64(tmp, cpu_env, offsetof(CPUState, spe_acc));

    tcg_temp_free_i64(tmp);

}
