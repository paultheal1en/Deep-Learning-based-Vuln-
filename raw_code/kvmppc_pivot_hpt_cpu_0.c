static void kvmppc_pivot_hpt_cpu(CPUState *cs, run_on_cpu_data arg)

{

    target_ulong sdr1 = arg.target_ptr;

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    CPUPPCState *env = &cpu->env;



    /* This is just for the benefit of PR KVM */

    cpu_synchronize_state(cs);

    env->spr[SPR_SDR1] = sdr1;

    if (kvmppc_put_books_sregs(cpu) < 0) {

        error_report("Unable to update SDR1 in KVM");

        exit(1);

    }

}
