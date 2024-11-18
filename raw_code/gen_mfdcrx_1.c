static void gen_mfdcrx(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        return;

    }

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    gen_helper_load_dcr(cpu_gpr[rD(ctx->opcode)], cpu_env,

                        cpu_gpr[rA(ctx->opcode)]);

    /* Note: Rc update flag set leads to undefined state of Rc0 */

#endif

}
