static void spapr_cpu_reset(void *opaque)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    PowerPCCPU *cpu = opaque;

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;



    cpu_reset(cs);



    /* All CPUs start halted.  CPU0 is unhalted from the machine level

     * reset code and the rest are explicitly started up by the guest

     * using an RTAS call */

    cs->halted = 1;



    env->spr[SPR_HIOR] = 0;



    ppc_hash64_set_external_hpt(cpu, spapr->htab, spapr->htab_shift,

                                &error_fatal);

}
