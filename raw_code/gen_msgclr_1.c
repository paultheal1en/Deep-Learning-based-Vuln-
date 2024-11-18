static void gen_msgclr(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    GEN_PRIV;

#else

    CHK_SV;

    gen_helper_msgclr(cpu_env, cpu_gpr[rB(ctx->opcode)]);

#endif /* defined(CONFIG_USER_ONLY) */

}
