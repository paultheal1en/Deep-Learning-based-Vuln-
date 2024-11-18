static void check_breakpoint(CPUState *env, DisasContext *dc)

{

    CPUBreakpoint *bp;



    if (unlikely(!TAILQ_EMPTY(&env->breakpoints))) {

        TAILQ_FOREACH(bp, &env->breakpoints, entry) {

            if (bp->pc == dc->pc) {

                t_gen_raise_exception(dc, EXCP_DEBUG);

                dc->is_jmp = DISAS_UPDATE;

             }

        }

    }

}
