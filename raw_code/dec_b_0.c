static void dec_b(DisasContext *dc)

{

    if (dc->r0 == R_RA) {

        LOG_DIS("ret\n");

    } else if (dc->r0 == R_EA) {

        LOG_DIS("eret\n");

    } else if (dc->r0 == R_BA) {

        LOG_DIS("bret\n");

    } else {

        LOG_DIS("b r%d\n", dc->r0);

    }



    /* restore IE.IE in case of an eret */

    if (dc->r0 == R_EA) {

        TCGv t0 = tcg_temp_new();

        int l1 = gen_new_label();

        tcg_gen_andi_tl(t0, cpu_ie, IE_EIE);

        tcg_gen_ori_tl(cpu_ie, cpu_ie, IE_IE);

        tcg_gen_brcondi_tl(TCG_COND_EQ, t0, IE_EIE, l1);

        tcg_gen_andi_tl(cpu_ie, cpu_ie, ~IE_IE);

        gen_set_label(l1);

        tcg_temp_free(t0);

    } else if (dc->r0 == R_BA) {

        TCGv t0 = tcg_temp_new();

        int l1 = gen_new_label();

        tcg_gen_andi_tl(t0, cpu_ie, IE_BIE);

        tcg_gen_ori_tl(cpu_ie, cpu_ie, IE_IE);

        tcg_gen_brcondi_tl(TCG_COND_EQ, t0, IE_BIE, l1);

        tcg_gen_andi_tl(cpu_ie, cpu_ie, ~IE_IE);

        gen_set_label(l1);

        tcg_temp_free(t0);

    }

    tcg_gen_mov_tl(cpu_pc, cpu_R[dc->r0]);



    dc->is_jmp = DISAS_JUMP;

}
