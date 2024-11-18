static void rtas_ibm_set_eeh_option(PowerPCCPU *cpu,

                                    sPAPRMachineState *spapr,

                                    uint32_t token, uint32_t nargs,

                                    target_ulong args, uint32_t nret,

                                    target_ulong rets)

{

    sPAPRPHBState *sphb;

    sPAPRPHBClass *spc;

    PCIDevice *pdev;

    uint32_t addr, option;

    uint64_t buid;

    int ret;



    if ((nargs != 4) || (nret != 1)) {

        goto param_error_exit;

    }



    buid = rtas_ldq(args, 1);

    addr = rtas_ld(args, 0);

    option = rtas_ld(args, 3);



    sphb = spapr_pci_find_phb(spapr, buid);

    if (!sphb) {

        goto param_error_exit;

    }



    pdev = pci_find_device(PCI_HOST_BRIDGE(sphb)->bus,

                           (addr >> 16) & 0xFF, (addr >> 8) & 0xFF);

    if (!pdev || !object_dynamic_cast(OBJECT(pdev), "vfio-pci")) {

        goto param_error_exit;

    }



    spc = SPAPR_PCI_HOST_BRIDGE_GET_CLASS(sphb);

    if (!spc->eeh_set_option) {

        goto param_error_exit;

    }



    ret = spc->eeh_set_option(sphb, addr, option);

    rtas_st(rets, 0, ret);

    return;



param_error_exit:

    rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

}
