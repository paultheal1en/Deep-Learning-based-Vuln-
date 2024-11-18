static void breakpoint_handler(CPUState *env)

{

    CPUBreakpoint *bp;



    if (env->watchpoint_hit) {

        if (env->watchpoint_hit->flags & BP_CPU) {

            env->watchpoint_hit = NULL;

            if (check_hw_breakpoints(env, 0))

                raise_exception(EXCP01_DB);

            else

                cpu_resume_from_signal(env, NULL);

        }

    } else {

        TAILQ_FOREACH(bp, &env->breakpoints, entry)

            if (bp->pc == env->eip) {

                if (bp->flags & BP_CPU) {

                    check_hw_breakpoints(env, 1);

                    raise_exception(EXCP01_DB);

                }

                break;

            }

    }

    if (prev_debug_excp_handler)

        prev_debug_excp_handler(env);

}
