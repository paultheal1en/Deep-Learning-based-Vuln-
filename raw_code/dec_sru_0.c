static void dec_sru(DisasContext *dc)

{

    if (dc->format == OP_FMT_RI) {

        LOG_DIS("srui r%d, r%d, %d\n", dc->r1, dc->r0, dc->imm5);

    } else {

        LOG_DIS("sru r%d, r%d, r%d\n", dc->r2, dc->r0, dc->r1);

    }



    if (dc->format == OP_FMT_RI) {

        if (!(dc->features & LM32_FEATURE_SHIFT) && (dc->imm5 != 1)) {

            qemu_log_mask(LOG_GUEST_ERROR,

                    "hardware shifter is not available\n");

            t_gen_illegal_insn(dc);

            return;

        }

        tcg_gen_shri_tl(cpu_R[dc->r1], cpu_R[dc->r0], dc->imm5);

    } else {

        int l1 = gen_new_label();

        int l2 = gen_new_label();

        TCGv t0 = tcg_temp_local_new();

        tcg_gen_andi_tl(t0, cpu_R[dc->r1], 0x1f);



        if (!(dc->features & LM32_FEATURE_SHIFT)) {

            tcg_gen_brcondi_tl(TCG_COND_EQ, t0, 1, l1);

            t_gen_illegal_insn(dc);

            tcg_gen_br(l2);

        }



        gen_set_label(l1);

        tcg_gen_shr_tl(cpu_R[dc->r2], cpu_R[dc->r0], t0);

        gen_set_label(l2);



        tcg_temp_free(t0);

    }

}
