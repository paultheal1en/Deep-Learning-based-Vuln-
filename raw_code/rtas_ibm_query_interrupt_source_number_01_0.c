static void rtas_ibm_query_interrupt_source_number(PowerPCCPU *cpu,

                                                   sPAPREnvironment *spapr,

                                                   uint32_t token,

                                                   uint32_t nargs,

                                                   target_ulong args,

                                                   uint32_t nret,

                                                   target_ulong rets)

{

    uint32_t config_addr = rtas_ld(args, 0);

    uint64_t buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    unsigned int intr_src_num = -1, ioa_intr_num = rtas_ld(args, 3);

    sPAPRPHBState *phb = NULL;

    PCIDevice *pdev = NULL;

    spapr_pci_msi *msi;



    /* Find sPAPRPHBState */

    phb = find_phb(spapr, buid);

    if (phb) {

        pdev = find_dev(spapr, buid, config_addr);

    }

    if (!phb || !pdev) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    /* Find device descriptor and start IRQ */

    msi = (spapr_pci_msi *) g_hash_table_lookup(phb->msi, &config_addr);

    if (!msi || !msi->first_irq || !msi->num || (ioa_intr_num >= msi->num)) {

        trace_spapr_pci_msi("Failed to return vector", config_addr);

        rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

        return;

    }

    intr_src_num = msi->first_irq + ioa_intr_num;

    trace_spapr_pci_rtas_ibm_query_interrupt_source_number(ioa_intr_num,

                                                           intr_src_num);



    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    rtas_st(rets, 1, intr_src_num);

    rtas_st(rets, 2, 1);/* 0 == level; 1 == edge */

}
