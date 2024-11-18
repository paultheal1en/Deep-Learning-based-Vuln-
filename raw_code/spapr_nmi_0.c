static void spapr_nmi(NMIState *n, int cpu_index, Error **errp)

{

    CPUState *cs;



    CPU_FOREACH(cs) {

        async_run_on_cpu(cs, ppc_cpu_do_nmi_on_cpu, RUN_ON_CPU_NULL);

    }

}
