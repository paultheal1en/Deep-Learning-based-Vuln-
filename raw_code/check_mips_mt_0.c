static always_inline void check_mips_mt(CPUState *env, DisasContext *ctx)

{

    if (unlikely(!(env->CP0_Config3 & (1 << CP0C3_MT))))

        generate_exception(ctx, EXCP_RI);

}
