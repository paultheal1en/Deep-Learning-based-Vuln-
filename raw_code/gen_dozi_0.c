static void gen_dozi(DisasContext *ctx)

{

    target_long simm = SIMM(ctx->opcode);

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    tcg_gen_brcondi_tl(TCG_COND_LT, cpu_gpr[rA(ctx->opcode)], simm, l1);

    tcg_gen_subfi_tl(cpu_gpr[rD(ctx->opcode)], simm, cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_tl(cpu_gpr[rD(ctx->opcode)], 0);

    gen_set_label(l2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

}
