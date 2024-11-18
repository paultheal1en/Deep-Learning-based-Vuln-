static void gen_dozo(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    TCGv t2 = tcg_temp_new();

    /* Start with XER OV disabled, the most likely case */

    tcg_gen_movi_tl(cpu_ov, 0);

    tcg_gen_brcond_tl(TCG_COND_GE, cpu_gpr[rB(ctx->opcode)], cpu_gpr[rA(ctx->opcode)], l1);

    tcg_gen_sub_tl(t0, cpu_gpr[rB(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_xor_tl(t1, cpu_gpr[rB(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_xor_tl(t2, cpu_gpr[rA(ctx->opcode)], t0);

    tcg_gen_andc_tl(t1, t1, t2);

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], t0);

    tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l2);

    tcg_gen_movi_tl(cpu_ov, 1);

    tcg_gen_movi_tl(cpu_so, 1);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_tl(cpu_gpr[rD(ctx->opcode)], 0);

    gen_set_label(l2);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    tcg_temp_free(t2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

}
