static void gen_loop(DisasContext *ctx, int r1, int32_t offset)

{

    int l1;

    l1 = gen_new_label();



    tcg_gen_subi_tl(cpu_gpr_a[r1], cpu_gpr_a[r1], 1);

    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_gpr_a[r1], -1, l1);

    gen_goto_tb(ctx, 1, ctx->pc + offset);

    gen_set_label(l1);

    gen_goto_tb(ctx, 0, ctx->next_pc);

}
