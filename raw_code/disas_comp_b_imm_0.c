static void disas_comp_b_imm(DisasContext *s, uint32_t insn)

{

    unsigned int sf, op, rt;

    uint64_t addr;

    int label_match;

    TCGv_i64 tcg_cmp;



    sf = extract32(insn, 31, 1);

    op = extract32(insn, 24, 1); /* 0: CBZ; 1: CBNZ */

    rt = extract32(insn, 0, 5);

    addr = s->pc + sextract32(insn, 5, 19) * 4 - 4;



    tcg_cmp = read_cpu_reg(s, rt, sf);

    label_match = gen_new_label();



    tcg_gen_brcondi_i64(op ? TCG_COND_NE : TCG_COND_EQ,

                        tcg_cmp, 0, label_match);



    gen_goto_tb(s, 0, s->pc);

    gen_set_label(label_match);

    gen_goto_tb(s, 1, addr);

}
