static void disas_cc(DisasContext *s, uint32_t insn)

{

    unsigned int sf, op, y, cond, rn, nzcv, is_imm;

    int label_continue = -1;

    TCGv_i64 tcg_tmp, tcg_y, tcg_rn;



    if (!extract32(insn, 29, 1)) {

        unallocated_encoding(s);

        return;

    }

    if (insn & (1 << 10 | 1 << 4)) {

        unallocated_encoding(s);

        return;

    }

    sf = extract32(insn, 31, 1);

    op = extract32(insn, 30, 1);

    is_imm = extract32(insn, 11, 1);

    y = extract32(insn, 16, 5); /* y = rm (reg) or imm5 (imm) */

    cond = extract32(insn, 12, 4);

    rn = extract32(insn, 5, 5);

    nzcv = extract32(insn, 0, 4);



    if (cond < 0x0e) { /* not always */

        int label_match = gen_new_label();

        label_continue = gen_new_label();

        arm_gen_test_cc(cond, label_match);

        /* nomatch: */

        tcg_tmp = tcg_temp_new_i64();

        tcg_gen_movi_i64(tcg_tmp, nzcv << 28);

        gen_set_nzcv(tcg_tmp);

        tcg_temp_free_i64(tcg_tmp);

        tcg_gen_br(label_continue);

        gen_set_label(label_match);

    }

    /* match, or condition is always */

    if (is_imm) {

        tcg_y = new_tmp_a64(s);

        tcg_gen_movi_i64(tcg_y, y);

    } else {

        tcg_y = cpu_reg(s, y);

    }

    tcg_rn = cpu_reg(s, rn);



    tcg_tmp = tcg_temp_new_i64();

    if (op) {

        gen_sub_CC(sf, tcg_tmp, tcg_rn, tcg_y);

    } else {

        gen_add_CC(sf, tcg_tmp, tcg_rn, tcg_y);

    }

    tcg_temp_free_i64(tcg_tmp);



    if (cond < 0x0e) { /* continue */

        gen_set_label(label_continue);

    }

}
