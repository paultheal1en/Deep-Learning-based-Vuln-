static void dec_scall(DisasContext *dc)

{

    if (dc->imm5 == 7) {

        LOG_DIS("scall\n");

    } else if (dc->imm5 == 2) {

        LOG_DIS("break\n");

    } else {

        cpu_abort(dc->env, "invalid opcode\n");

    }



    if (dc->imm5 == 7) {

        tcg_gen_movi_tl(cpu_pc, dc->pc);

        t_gen_raise_exception(dc, EXCP_SYSTEMCALL);

    } else {

        tcg_gen_movi_tl(cpu_pc, dc->pc);

        t_gen_raise_exception(dc, EXCP_BREAKPOINT);

    }

}
