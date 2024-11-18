static void rtas_ibm_get_config_addr_info2(PowerPCCPU *cpu,

                                           sPAPREnvironment *spapr,

                                           uint32_t token, uint32_t nargs,

                                           target_ulong args, uint32_t nret,

                                           target_ulong rets)

{

    sPAPRPHBState *sphb;

    sPAPRPHBClass *spc;

    PCIDevice *pdev;

    uint32_t addr, option;

    uint64_t buid;



    if ((nargs != 4) || (nret != 2)) {

        goto param_error_exit;

    }



    buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    sphb = find_phb(spapr, buid);

    if (!sphb) {

        goto param_error_exit;

    }



    spc = SPAPR_PCI_HOST_BRIDGE_GET_CLASS(sphb);

    if (!spc->eeh_set_option) {

        goto param_error_exit;

    }



    /*

     * We always have PE address of form "00BB0001". "BB"

     * represents the bus number of PE's primary bus.

     */

    option = rtas_ld(args, 3);

    switch (option) {

    case RTAS_GET_PE_ADDR:

        addr = rtas_ld(args, 0);

        pdev = find_dev(spapr, buid, addr);

        if (!pdev) {

            goto param_error_exit;

        }



        rtas_st(rets, 1, (pci_bus_num(pdev->bus) << 16) + 1);

        break;

    case RTAS_GET_PE_MODE:

        rtas_st(rets, 1, RTAS_PE_MODE_SHARED);

        break;

    default:

        goto param_error_exit;

    }



    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    return;



param_error_exit:

    rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

}
