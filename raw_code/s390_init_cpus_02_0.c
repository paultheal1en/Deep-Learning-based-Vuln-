static void s390_init_cpus(MachineState *machine)

{

    MachineClass *mc = MACHINE_GET_CLASS(machine);

    int i;



    if (tcg_enabled() && max_cpus > 1) {

        error_report("Number of SMP CPUs requested (%d) exceeds max CPUs "

                     "supported by TCG (1) on s390x", max_cpus);

        exit(1);

    }



    /* initialize possible_cpus */

    mc->possible_cpu_arch_ids(machine);



    for (i = 0; i < smp_cpus; i++) {

        s390x_new_cpu(machine->cpu_type, i, &error_fatal);

    }

}
