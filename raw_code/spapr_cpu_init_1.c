static void spapr_cpu_init(sPAPRMachineState *spapr, PowerPCCPU *cpu,

                           Error **errp)

{

    CPUPPCState *env = &cpu->env;

    CPUState *cs = CPU(cpu);

    int i;



    /* Set time-base frequency to 512 MHz */

    cpu_ppc_tb_init(env, SPAPR_TIMEBASE_FREQ);



    /* Enable PAPR mode in TCG or KVM */


    cpu_ppc_set_papr(cpu);



    if (cpu->max_compat) {

        Error *local_err = NULL;



        ppc_set_compat(cpu, cpu->max_compat, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

    }



    /* Set NUMA node for the added CPUs  */

    i = numa_get_node_for_cpu(cs->cpu_index);

    if (i < nb_numa_nodes) {

            cs->numa_node = i;

    }



    xics_cpu_setup(spapr->xics, cpu);



    qemu_register_reset(spapr_cpu_reset, cpu);

    spapr_cpu_reset(cpu);

}