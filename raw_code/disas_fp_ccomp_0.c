static void disas_fp_ccomp(DisasContext *s, uint32_t insn)

{

    unsigned int mos, type, rm, cond, rn, op, nzcv;

    TCGv_i64 tcg_flags;

    int label_continue = -1;



    mos = extract32(insn, 29, 3);

    type = extract32(insn, 22, 2); /* 0 = single, 1 = double */

    rm = extract32(insn, 16, 5);

    cond = extract32(insn, 12, 4);

    rn = extract32(insn, 5, 5);

    op = extract32(insn, 4, 1);

    nzcv = extract32(insn, 0, 4);



    if (mos || type > 1) {

        unallocated_encoding(s);

        return;

    }



    if (!fp_access_check(s)) {

        return;

    }



    if (cond < 0x0e) { /* not always */

        int label_match = gen_new_label();

        label_continue = gen_new_label();

        arm_gen_test_cc(cond, label_match);

        /* nomatch: */

        tcg_flags = tcg_const_i64(nzcv << 28);

        gen_set_nzcv(tcg_flags);

        tcg_temp_free_i64(tcg_flags);

        tcg_gen_br(label_continue);

        gen_set_label(label_match);

    }



    handle_fp_compare(s, type, rn, rm, false, op);



    if (cond < 0x0e) {

        gen_set_label(label_continue);

    }

}
