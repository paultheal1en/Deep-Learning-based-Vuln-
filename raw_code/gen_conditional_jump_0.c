static void gen_conditional_jump(DisasContext * ctx,

				 target_ulong ift, target_ulong ifnott)

{

    int l1;

    TCGv sr;



    l1 = gen_new_label();

    sr = tcg_temp_new();

    tcg_gen_andi_i32(sr, cpu_sr, SR_T);

    tcg_gen_brcondi_i32(TCG_COND_NE, sr, 0, l1);

    gen_goto_tb(ctx, 0, ifnott);

    gen_set_label(l1);

    gen_goto_tb(ctx, 1, ift);

}
