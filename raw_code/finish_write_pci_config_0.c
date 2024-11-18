static void finish_write_pci_config(sPAPREnvironment *spapr, uint64_t buid,

                                    uint32_t addr, uint32_t size,

                                    uint32_t val, target_ulong rets)

{

    PCIDevice *pci_dev;



    if ((size != 1) && (size != 2) && (size != 4)) {

        /* access must be 1, 2 or 4 bytes */

        rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

        return;

    }



    pci_dev = find_dev(spapr, buid, addr);

    addr = rtas_pci_cfgaddr(addr);



    if (!pci_dev || (addr % size) || (addr >= pci_config_size(pci_dev))) {

        /* Access must be to a valid device, within bounds and

         * naturally aligned */

        rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

        return;

    }



    pci_host_config_write_common(pci_dev, addr, pci_config_size(pci_dev),

                                 val, size);



    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

}
