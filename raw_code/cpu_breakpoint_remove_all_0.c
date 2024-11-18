void cpu_breakpoint_remove_all(CPUState *cpu, int mask)

{

#if defined(TARGET_HAS_ICE)

    CPUBreakpoint *bp, *next;



    QTAILQ_FOREACH_SAFE(bp, &cpu->breakpoints, entry, next) {

        if (bp->flags & mask) {

            cpu_breakpoint_remove_by_ref(cpu, bp);

        }

    }

#endif

}
