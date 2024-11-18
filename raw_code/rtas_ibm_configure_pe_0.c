static void rtas_ibm_configure_pe(PowerPCCPU *cpu,

                                  sPAPREnvironment *spapr,

                                  uint32_t token, uint32_t nargs,

                                  target_ulong args, uint32_t nret,

                                  target_ulong rets)

{

    sPAPRPHBState *sphb;

    sPAPRPHBClass *spc;

    uint64_t buid;

    int ret;



    if ((nargs != 3) || (nret != 1)) {

        goto param_error_exit;

    }



    buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    sphb = find_phb(spapr, buid);

    if (!sphb) {

        goto param_error_exit;

    }



    spc = SPAPR_PCI_HOST_BRIDGE_GET_CLASS(sphb);

    if (!spc->eeh_configure) {

        goto param_error_exit;

    }



    ret = spc->eeh_configure(sphb);

    rtas_st(rets, 0, ret);

    return;



param_error_exit:

    rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

}
