static void disas_test_b_imm(DisasContext *s, uint32_t insn)

{

    unsigned int bit_pos, op, rt;

    uint64_t addr;

    int label_match;

    TCGv_i64 tcg_cmp;



    bit_pos = (extract32(insn, 31, 1) << 5) | extract32(insn, 19, 5);

    op = extract32(insn, 24, 1); /* 0: TBZ; 1: TBNZ */

    addr = s->pc + sextract32(insn, 5, 14) * 4 - 4;

    rt = extract32(insn, 0, 5);



    tcg_cmp = tcg_temp_new_i64();

    tcg_gen_andi_i64(tcg_cmp, cpu_reg(s, rt), (1ULL << bit_pos));

    label_match = gen_new_label();

    tcg_gen_brcondi_i64(op ? TCG_COND_NE : TCG_COND_EQ,

                        tcg_cmp, 0, label_match);

    tcg_temp_free_i64(tcg_cmp);

    gen_goto_tb(s, 0, s->pc);

    gen_set_label(label_match);

    gen_goto_tb(s, 1, addr);

}
