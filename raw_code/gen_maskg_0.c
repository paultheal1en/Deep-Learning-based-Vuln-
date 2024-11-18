static void gen_maskg(DisasContext *ctx)

{

    int l1 = gen_new_label();

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    TCGv t2 = tcg_temp_new();

    TCGv t3 = tcg_temp_new();

    tcg_gen_movi_tl(t3, 0xFFFFFFFF);

    tcg_gen_andi_tl(t0, cpu_gpr[rB(ctx->opcode)], 0x1F);

    tcg_gen_andi_tl(t1, cpu_gpr[rS(ctx->opcode)], 0x1F);

    tcg_gen_addi_tl(t2, t0, 1);

    tcg_gen_shr_tl(t2, t3, t2);

    tcg_gen_shr_tl(t3, t3, t1);

    tcg_gen_xor_tl(cpu_gpr[rA(ctx->opcode)], t2, t3);

    tcg_gen_brcond_tl(TCG_COND_GE, t0, t1, l1);

    tcg_gen_neg_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    gen_set_label(l1);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    tcg_temp_free(t2);

    tcg_temp_free(t3);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
