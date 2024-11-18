static target_ulong h_put_tce_indirect(PowerPCCPU *cpu,

                                       sPAPRMachineState *spapr,

                                       target_ulong opcode, target_ulong *args)

{

    int i;

    target_ulong liobn = args[0];

    target_ulong ioba = args[1];

    target_ulong ioba1 = ioba;

    target_ulong tce_list = args[2];

    target_ulong npages = args[3];

    target_ulong ret = H_PARAMETER, tce = 0;

    sPAPRTCETable *tcet = spapr_tce_find_by_liobn(liobn);

    CPUState *cs = CPU(cpu);

    hwaddr page_mask, page_size;



    if (!tcet) {

        return H_PARAMETER;

    }



    if ((npages > 512) || (tce_list & SPAPR_TCE_PAGE_MASK)) {

        return H_PARAMETER;

    }



    page_mask = IOMMU_PAGE_MASK(tcet->page_shift);

    page_size = IOMMU_PAGE_SIZE(tcet->page_shift);

    ioba &= page_mask;



    for (i = 0; i < npages; ++i, ioba += page_size) {

        target_ulong off = (tce_list & ~SPAPR_TCE_RW) +

                                i * sizeof(target_ulong);

        tce = ldq_be_phys(cs->as, off);



        ret = put_tce_emu(tcet, ioba, tce);

        if (ret) {

            break;

        }

    }



    /* Trace last successful or the first problematic entry */

    i = i ? (i - 1) : 0;

    if (SPAPR_IS_PCI_LIOBN(liobn)) {

        trace_spapr_iommu_pci_indirect(liobn, ioba1, tce_list, i, tce, ret);

    } else {

        trace_spapr_iommu_indirect(liobn, ioba1, tce_list, i, tce, ret);

    }

    return ret;

}
