static void aarch64_tr_translate_insn(DisasContextBase *dcbase, CPUState *cpu)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);

    CPUARMState *env = cpu->env_ptr;



    if (dc->ss_active && !dc->pstate_ss) {

        /* Singlestep state is Active-pending.

         * If we're in this state at the start of a TB then either

         *  a) we just took an exception to an EL which is being debugged

         *     and this is the first insn in the exception handler

         *  b) debug exceptions were masked and we just unmasked them

         *     without changing EL (eg by clearing PSTATE.D)

         * In either case we're going to take a swstep exception in the

         * "did not step an insn" case, and so the syndrome ISV and EX

         * bits should be zero.

         */

        assert(dc->base.num_insns == 1);

        gen_exception(EXCP_UDEF, syn_swstep(dc->ss_same_el, 0, 0),

                      default_exception_el(dc));

        dc->base.is_jmp = DISAS_NORETURN;

    } else {

        disas_a64_insn(env, dc);

    }



    if (dc->base.is_jmp == DISAS_NEXT) {

        if (dc->ss_active || dc->pc >= dc->next_page_start) {

            dc->base.is_jmp = DISAS_TOO_MANY;

        }

    }



    dc->base.pc_next = dc->pc;

    translator_loop_temp_check(&dc->base);

}
