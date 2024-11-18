static int spapr_msicfg_find(sPAPRPHBState *phb, uint32_t config_addr,

                             bool alloc_new)

{

    int i;



    for (i = 0; i < SPAPR_MSIX_MAX_DEVS; ++i) {

        if (!phb->msi_table[i].nvec) {

            break;

        }

        if (phb->msi_table[i].config_addr == config_addr) {

            return i;

        }

    }

    if ((i < SPAPR_MSIX_MAX_DEVS) && alloc_new) {

        trace_spapr_pci_msi("Allocating new MSI config", i, config_addr);

        return i;

    }



    return -1;

}
