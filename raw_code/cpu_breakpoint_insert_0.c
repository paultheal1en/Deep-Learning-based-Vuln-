int cpu_breakpoint_insert(CPUState *cpu, vaddr pc, int flags,

                          CPUBreakpoint **breakpoint)

{

#if defined(TARGET_HAS_ICE)

    CPUBreakpoint *bp;



    bp = g_malloc(sizeof(*bp));



    bp->pc = pc;

    bp->flags = flags;



    /* keep all GDB-injected breakpoints in front */

    if (flags & BP_GDB) {

        QTAILQ_INSERT_HEAD(&cpu->breakpoints, bp, entry);

    } else {

        QTAILQ_INSERT_TAIL(&cpu->breakpoints, bp, entry);

    }



    breakpoint_invalidate(cpu, pc);



    if (breakpoint) {

        *breakpoint = bp;

    }

    return 0;

#else

    return -ENOSYS;

#endif

}
