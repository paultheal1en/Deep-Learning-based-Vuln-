static void gen_eob_inhibit_irq(DisasContext *s, bool inhibit)

{

    gen_update_cc_op(s);



    /* If several instructions disable interrupts, only the first does it.  */

    if (inhibit && !(s->flags & HF_INHIBIT_IRQ_MASK)) {

        gen_set_hflag(s, HF_INHIBIT_IRQ_MASK);

    } else {

        gen_reset_hflag(s, HF_INHIBIT_IRQ_MASK);

    }



    if (s->tb->flags & HF_RF_MASK) {

        gen_helper_reset_rf(cpu_env);

    }

    if (s->singlestep_enabled) {

        gen_helper_debug(cpu_env);

    } else if (s->tf) {

        gen_helper_single_step(cpu_env);

    } else {

        tcg_gen_exit_tb(0);

    }

    s->is_jmp = DISAS_TB_JUMP;

}
