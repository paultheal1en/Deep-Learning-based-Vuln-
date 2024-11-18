static void breakpoint_invalidate(CPUState *cpu, target_ulong pc)

{

    /* Flush the whole TB as this will not have race conditions

     * even if we don't have proper locking yet.

     * Ideally we would just invalidate the TBs for the

     * specified PC.

     */

    tb_flush(cpu);

}
