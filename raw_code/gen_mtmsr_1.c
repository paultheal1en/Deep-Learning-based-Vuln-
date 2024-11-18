static void gen_mtmsr(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        return;

    }

    if (ctx->opcode & 0x00010000) {

        /* Special form that does not need any synchronisation */

        TCGv t0 = tcg_temp_new();

        tcg_gen_andi_tl(t0, cpu_gpr[rS(ctx->opcode)], (1 << MSR_RI) | (1 << MSR_EE));

        tcg_gen_andi_tl(cpu_msr, cpu_msr, ~(target_ulong)((1 << MSR_RI) | (1 << MSR_EE)));

        tcg_gen_or_tl(cpu_msr, cpu_msr, t0);

        tcg_temp_free(t0);

    } else {

        TCGv msr = tcg_temp_new();



        /* XXX: we need to update nip before the store

         *      if we enter power saving mode, we will exit the loop

         *      directly from ppc_store_msr

         */

        gen_update_nip(ctx, ctx->nip);

#if defined(TARGET_PPC64)

        tcg_gen_deposit_tl(msr, cpu_msr, cpu_gpr[rS(ctx->opcode)], 0, 32);

#else

        tcg_gen_mov_tl(msr, cpu_gpr[rS(ctx->opcode)]);

#endif

        gen_helper_store_msr(cpu_env, msr);

        tcg_temp_free(msr);

        /* Must stop the translation as machine state (may have) changed */

        /* Note that mtmsr is not always defined as context-synchronizing */

        gen_stop_exception(ctx);

    }

#endif

}
