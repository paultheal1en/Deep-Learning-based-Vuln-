static void gen_delayed_conditional_jump(DisasContext * ctx)

{

    int l1;

    TCGv ds;



    l1 = gen_new_label();

    ds = tcg_temp_new();

    tcg_gen_andi_i32(ds, cpu_flags, DELAY_SLOT_TRUE);

    tcg_gen_brcondi_i32(TCG_COND_NE, ds, 0, l1);

    gen_goto_tb(ctx, 1, ctx->pc + 2);

    gen_set_label(l1);

    tcg_gen_andi_i32(cpu_flags, cpu_flags, ~DELAY_SLOT_TRUE);

    gen_jump(ctx);

}
