static inline void gen_evmwumiaa(DisasContext *ctx)

{

    TCGv_i64 acc;

    TCGv_i64 tmp;



    if (unlikely(!ctx->spe_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_APU);

        return;

    }



    gen_evmwumi(ctx);           /* rD := rA * rB */



    acc = tcg_temp_new_i64();

    tmp = tcg_temp_new_i64();



    /* tmp := rD */

    gen_load_gpr64(tmp, rD(ctx->opcode));



    /* Load acc */

    tcg_gen_ld_i64(acc, cpu_env, offsetof(CPUState, spe_acc));



    /* acc := tmp + acc */

    tcg_gen_add_i64(acc, acc, tmp);



    /* Store acc */

    tcg_gen_st_i64(acc, cpu_env, offsetof(CPUState, spe_acc));



    /* rD := acc */

    gen_store_gpr64(rD(ctx->opcode), acc);



    tcg_temp_free_i64(acc);

    tcg_temp_free_i64(tmp);

}
