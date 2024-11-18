static int timebase_post_load(void *opaque, int version_id)

{

    PPCTimebase *tb_remote = opaque;

    CPUState *cpu;

    PowerPCCPU *first_ppc_cpu = POWERPC_CPU(first_cpu);

    int64_t tb_off_adj, tb_off, ns_diff;

    int64_t migration_duration_ns, migration_duration_tb, guest_tb, host_ns;

    unsigned long freq;



    if (!first_ppc_cpu->env.tb_env) {

        error_report("No timebase object");

        return -1;

    }



    freq = first_ppc_cpu->env.tb_env->tb_freq;

    /*

     * Calculate timebase on the destination side of migration.

     * The destination timebase must be not less than the source timebase.

     * We try to adjust timebase by downtime if host clocks are not

     * too much out of sync (1 second for now).

     */

    host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

    ns_diff = MAX(0, host_ns - tb_remote->time_of_the_day_ns);

    migration_duration_ns = MIN(NANOSECONDS_PER_SECOND, ns_diff);

    migration_duration_tb = muldiv64(migration_duration_ns, freq,

                                     NANOSECONDS_PER_SECOND);

    guest_tb = tb_remote->guest_timebase + MIN(0, migration_duration_tb);



    tb_off_adj = guest_tb - cpu_get_host_ticks();



    tb_off = first_ppc_cpu->env.tb_env->tb_offset;

    trace_ppc_tb_adjust(tb_off, tb_off_adj, tb_off_adj - tb_off,

                        (tb_off_adj - tb_off) / freq);



    /* Set new offset to all CPUs */

    CPU_FOREACH(cpu) {

        PowerPCCPU *pcpu = POWERPC_CPU(cpu);

        pcpu->env.tb_env->tb_offset = tb_off_adj;

    }



    return 0;

}
