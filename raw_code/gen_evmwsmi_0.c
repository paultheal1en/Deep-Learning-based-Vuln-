static inline void gen_evmwsmi(DisasContext *ctx)

{

    TCGv_i64 t0, t1;



    if (unlikely(!ctx->spe_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_APU);

        return;

    }



    t0 = tcg_temp_new_i64();

    t1 = tcg_temp_new_i64();



    /* t0 := rA; t1 := rB */

#if defined(TARGET_PPC64)

    tcg_gen_ext32s_tl(t0, cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_ext32s_tl(t1, cpu_gpr[rB(ctx->opcode)]);

#else

    tcg_gen_ext_tl_i64(t0, cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_ext_tl_i64(t1, cpu_gpr[rB(ctx->opcode)]);

#endif



    tcg_gen_mul_i64(t0, t0, t1);  /* t0 := rA * rB */



    gen_store_gpr64(rD(ctx->opcode), t0); /* rD := t0 */



    tcg_temp_free_i64(t0);

    tcg_temp_free_i64(t1);

}
