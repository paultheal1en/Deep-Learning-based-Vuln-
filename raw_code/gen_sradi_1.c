static always_inline void gen_sradi (DisasContext *ctx, int n)

{

    int sh = SH(ctx->opcode) + (n << 5);

    if (sh != 0) {

        int l1, l2;

        TCGv t0;

        l1 = gen_new_label();

        l2 = gen_new_label();

        tcg_gen_brcondi_tl(TCG_COND_GE, cpu_gpr[rS(ctx->opcode)], 0, l1);

        t0 = tcg_temp_new(TCG_TYPE_TL);

        tcg_gen_andi_tl(t0, cpu_gpr[rS(ctx->opcode)], (1ULL << sh) - 1);

        tcg_gen_brcondi_tl(TCG_COND_EQ, t0, 0, l1);

        tcg_gen_ori_tl(cpu_xer, cpu_xer, 1 << XER_CA);

        tcg_gen_br(l2);

        gen_set_label(l1);

        tcg_gen_andi_tl(cpu_xer, cpu_xer, ~(1 << XER_CA));

        gen_set_label(l2);

        tcg_gen_sari_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rS(ctx->opcode)], sh);

    } else {

        tcg_gen_mov_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rS(ctx->opcode)]);

        tcg_gen_andi_tl(cpu_xer, cpu_xer, ~(1 << XER_CA));

    }

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
