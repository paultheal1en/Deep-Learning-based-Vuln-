static inline void gen_cond_branch(DisasContext *dc, int cond)

{

    int l1;



    l1 = gen_new_label();

    tcg_gen_brcond_tl(cond, cpu_R[dc->r0], cpu_R[dc->r1], l1);

    gen_goto_tb(dc, 0, dc->pc + 4);

    gen_set_label(l1);

    gen_goto_tb(dc, 1, dc->pc + (sign_extend(dc->imm16 << 2, 16)));

    dc->is_jmp = DISAS_TB_JUMP;

}
