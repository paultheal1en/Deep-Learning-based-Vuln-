static void disas_fp_csel(DisasContext *s, uint32_t insn)

{

    unsigned int mos, type, rm, cond, rn, rd;

    int label_continue = -1;



    mos = extract32(insn, 29, 3);

    type = extract32(insn, 22, 2); /* 0 = single, 1 = double */

    rm = extract32(insn, 16, 5);

    cond = extract32(insn, 12, 4);

    rn = extract32(insn, 5, 5);

    rd = extract32(insn, 0, 5);



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

        gen_mov_fp2fp(s, type, rd, rm);

        tcg_gen_br(label_continue);

        gen_set_label(label_match);

    }



    gen_mov_fp2fp(s, type, rd, rn);



    if (cond < 0x0e) { /* continue */

        gen_set_label(label_continue);

    }

}
