static int dec_scc_r(CPUCRISState *env, DisasContext *dc)

{

    int cond = dc->op2;



    LOG_DIS("s%s $r%u\n",

            cc_name(cond), dc->op1);



    if (cond != CC_A) {

        int l1;



        gen_tst_cc(dc, cpu_R[dc->op1], cond);

        l1 = gen_new_label();

        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_R[dc->op1], 0, l1);

        tcg_gen_movi_tl(cpu_R[dc->op1], 1);

        gen_set_label(l1);

    } else {

        tcg_gen_movi_tl(cpu_R[dc->op1], 1);

    }



    cris_cc_mask(dc, 0);

    return 2;

}
