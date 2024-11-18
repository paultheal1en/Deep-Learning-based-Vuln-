static void rtas_start_cpu(sPAPREnvironment *spapr,

                           uint32_t token, uint32_t nargs,

                           target_ulong args,

                           uint32_t nret, target_ulong rets)

{

    target_ulong id, start, r3;

    CPUState *cs;



    if (nargs != 3 || nret != 1) {

        rtas_st(rets, 0, -3);

        return;

    }



    id = rtas_ld(args, 0);

    start = rtas_ld(args, 1);

    r3 = rtas_ld(args, 2);



    cs = qemu_get_cpu(id);

    if (cs != NULL) {

        PowerPCCPU *cpu = POWERPC_CPU(cs);

        CPUPPCState *env = &cpu->env;



        if (!cs->halted) {

            rtas_st(rets, 0, -1);

            return;

        }



        /* This will make sure qemu state is up to date with kvm, and

         * mark it dirty so our changes get flushed back before the

         * new cpu enters */

        kvm_cpu_synchronize_state(cs);



        env->msr = (1ULL << MSR_SF) | (1ULL << MSR_ME);

        env->nip = start;

        env->gpr[3] = r3;

        cs->halted = 0;



        qemu_cpu_kick(cs);



        rtas_st(rets, 0, 0);

        return;

    }



    /* Didn't find a matching cpu */

    rtas_st(rets, 0, -3);

}
