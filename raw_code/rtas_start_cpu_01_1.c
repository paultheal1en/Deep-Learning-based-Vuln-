static void rtas_start_cpu(PowerPCCPU *cpu_, sPAPRMachineState *spapr,

                           uint32_t token, uint32_t nargs,

                           target_ulong args,

                           uint32_t nret, target_ulong rets)

{

    target_ulong id, start, r3;

    PowerPCCPU *cpu;



    if (nargs != 3 || nret != 1) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    id = rtas_ld(args, 0);

    start = rtas_ld(args, 1);

    r3 = rtas_ld(args, 2);



    cpu = spapr_find_cpu(id);

    if (cpu != NULL) {

        CPUState *cs = CPU(cpu);

        CPUPPCState *env = &cpu->env;




        if (!cs->halted) {

            rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

            return;

        }



        /* This will make sure qemu state is up to date with kvm, and

         * mark it dirty so our changes get flushed back before the

         * new cpu enters */

        kvm_cpu_synchronize_state(cs);



        env->msr = (1ULL << MSR_SF) | (1ULL << MSR_ME);



        /* Enable Power-saving mode Exit Cause exceptions for the new CPU */

        env->spr[SPR_LPCR] |= pcc->lpcr_pm;



        env->nip = start;

        env->gpr[3] = r3;

        cs->halted = 0;

        spapr_cpu_set_endianness(cpu);

        spapr_cpu_update_tb_offset(cpu);



        qemu_cpu_kick(cs);



        rtas_st(rets, 0, RTAS_OUT_SUCCESS);

        return;

    }



    /* Didn't find a matching cpu */

    rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

}