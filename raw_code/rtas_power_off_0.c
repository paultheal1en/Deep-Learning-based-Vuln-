static void rtas_power_off(sPAPREnvironment *spapr,

                           uint32_t token, uint32_t nargs, target_ulong args,

                           uint32_t nret, target_ulong rets)

{

    if (nargs != 2 || nret != 1) {

        rtas_st(rets, 0, -3);

        return;

    }

    qemu_system_shutdown_request();

    rtas_st(rets, 0, 0);

}
