int cpu_breakpoint_insert(CPUState *env, target_ulong pc, int flags,

                          CPUBreakpoint **breakpoint)

{

#if defined(TARGET_HAS_ICE)

    CPUBreakpoint *bp;



    bp = qemu_malloc(sizeof(*bp));



    bp->pc = pc;

    bp->flags = flags;



    /* keep all GDB-injected breakpoints in front */

    if (flags & BP_GDB)

        TAILQ_INSERT_HEAD(&env->breakpoints, bp, entry);

    else

        TAILQ_INSERT_TAIL(&env->breakpoints, bp, entry);



    breakpoint_invalidate(env, pc);



    if (breakpoint)

        *breakpoint = bp;

    return 0;

#else

    return -ENOSYS;

#endif

}
