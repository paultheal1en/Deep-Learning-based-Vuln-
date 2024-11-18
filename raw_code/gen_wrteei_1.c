static void gen_wrteei(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    if (ctx->opcode & 0x00008000) {

        tcg_gen_ori_tl(cpu_msr, cpu_msr, (1 << MSR_EE));

        /* Stop translation to have a chance to raise an exception */

        gen_stop_exception(ctx);

    } else {

        tcg_gen_andi_tl(cpu_msr, cpu_msr, ~(1 << MSR_EE));

    }

#endif

}
