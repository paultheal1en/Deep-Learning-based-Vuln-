void HELPER(yield)(CPUARMState *env)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    CPUState *cs = CPU(cpu);



    /* When running in MTTCG we don't generate jumps to the yield and

     * WFE helpers as it won't affect the scheduling of other vCPUs.

     * If we wanted to more completely model WFE/SEV so we don't busy

     * spin unnecessarily we would need to do something more involved.

     */

    g_assert(!parallel_cpus);



    /* This is a non-trappable hint instruction that generally indicates

     * that the guest is currently busy-looping. Yield control back to the

     * top level loop so that a more deserving VCPU has a chance to run.

     */

    cs->exception_index = EXCP_YIELD;

    cpu_loop_exit(cs);

}
