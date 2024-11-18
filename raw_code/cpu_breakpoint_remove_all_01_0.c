void cpu_breakpoint_remove_all(CPUState *env, int mask)

{

#if defined(TARGET_HAS_ICE)

    CPUBreakpoint *bp, *next;



    TAILQ_FOREACH_SAFE(bp, &env->breakpoints, entry, next) {

        if (bp->flags & mask)

            cpu_breakpoint_remove_by_ref(env, bp);

    }

#endif

}
