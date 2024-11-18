static void rtas_query_cpu_stopped_state(sPAPREnvironment *spapr,

                                         uint32_t token, uint32_t nargs,

                                         target_ulong args,

                                         uint32_t nret, target_ulong rets)

{

    target_ulong id;

    CPUState *cpu;



    if (nargs != 1 || nret != 2) {

        rtas_st(rets, 0, -3);

        return;

    }



    id = rtas_ld(args, 0);

    cpu = qemu_get_cpu(id);

    if (cpu != NULL) {

        if (cpu->halted) {

            rtas_st(rets, 1, 0);

        } else {

            rtas_st(rets, 1, 2);

        }



        rtas_st(rets, 0, 0);

        return;

    }



    /* Didn't find a matching cpu */

    rtas_st(rets, 0, -3);

}
