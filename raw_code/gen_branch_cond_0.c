static inline void gen_branch_cond(DisasContext *ctx, TCGCond cond, TCGv r1,

                                   TCGv r2, int16_t address)

{

    int jumpLabel;

    jumpLabel = gen_new_label();

    tcg_gen_brcond_tl(cond, r1, r2, jumpLabel);



    gen_goto_tb(ctx, 1, ctx->next_pc);



    gen_set_label(jumpLabel);

    gen_goto_tb(ctx, 0, ctx->pc + address * 2);

}
