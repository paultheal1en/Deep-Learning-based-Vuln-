static void disas_cond_b_imm(DisasContext *s, uint32_t insn)

{

    unsigned int cond;

    uint64_t addr;



    if ((insn & (1 << 4)) || (insn & (1 << 24))) {

        unallocated_encoding(s);

        return;

    }

    addr = s->pc + sextract32(insn, 5, 19) * 4 - 4;

    cond = extract32(insn, 0, 4);



    if (cond < 0x0e) {

        /* genuinely conditional branches */

        int label_match = gen_new_label();

        arm_gen_test_cc(cond, label_match);

        gen_goto_tb(s, 0, s->pc);

        gen_set_label(label_match);

        gen_goto_tb(s, 1, addr);

    } else {

        /* 0xe and 0xf are both "always" conditions */

        gen_goto_tb(s, 0, addr);

    }

}
