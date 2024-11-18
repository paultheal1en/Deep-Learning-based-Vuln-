static void gen_bxx(DisasContext *dc, uint32_t code, uint32_t flags)

{

    I_TYPE(instr, code);



    TCGLabel *l1 = gen_new_label();

    tcg_gen_brcond_tl(flags, dc->cpu_R[instr.a], dc->cpu_R[instr.b], l1);

    gen_goto_tb(dc, 0, dc->pc + 4);

    gen_set_label(l1);

    gen_goto_tb(dc, 1, dc->pc + 4 + (instr.imm16s & -4));

    dc->is_jmp = DISAS_TB_JUMP;

}
