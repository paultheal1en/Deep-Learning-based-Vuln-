static void rtas_ibm_read_slot_reset_state2(PowerPCCPU *cpu,

                                            sPAPREnvironment *spapr,

                                            uint32_t token, uint32_t nargs,

                                            target_ulong args, uint32_t nret,

                                            target_ulong rets)

{

    sPAPRPHBState *sphb;

    sPAPRPHBClass *spc;

    uint64_t buid;

    int state, ret;



    if ((nargs != 3) || (nret != 4 && nret != 5)) {

        goto param_error_exit;

    }



    buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    sphb = find_phb(spapr, buid);

    if (!sphb) {

        goto param_error_exit;

    }



    spc = SPAPR_PCI_HOST_BRIDGE_GET_CLASS(sphb);

    if (!spc->eeh_get_state) {

        goto param_error_exit;

    }



    ret = spc->eeh_get_state(sphb, &state);

    rtas_st(rets, 0, ret);

    if (ret != RTAS_OUT_SUCCESS) {

        return;

    }



    rtas_st(rets, 1, state);

    rtas_st(rets, 2, RTAS_EEH_SUPPORT);

    rtas_st(rets, 3, RTAS_EEH_PE_UNAVAIL_INFO);

    if (nret >= 5) {

        rtas_st(rets, 4, RTAS_EEH_PE_RECOVER_INFO);

    }

    return;



param_error_exit:

    rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

}
