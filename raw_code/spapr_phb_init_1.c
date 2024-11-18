static int spapr_phb_init(SysBusDevice *s)

{

    sPAPRPHBState *sphb = SPAPR_PCI_HOST_BRIDGE(s);

    PCIHostState *phb = PCI_HOST_BRIDGE(s);

    char *namebuf;

    int i;

    PCIBus *bus;



    sphb->dtbusname = g_strdup_printf("pci@%" PRIx64, sphb->buid);

    namebuf = alloca(strlen(sphb->dtbusname) + 32);



    /* Initialize memory regions */

    sprintf(namebuf, "%s.mmio", sphb->dtbusname);

    memory_region_init(&sphb->memspace, namebuf, INT64_MAX);



    sprintf(namebuf, "%s.mmio-alias", sphb->dtbusname);

    memory_region_init_alias(&sphb->memwindow, namebuf, &sphb->memspace,

                             SPAPR_PCI_MEM_WIN_BUS_OFFSET, sphb->mem_win_size);

    memory_region_add_subregion(get_system_memory(), sphb->mem_win_addr,

                                &sphb->memwindow);



    /* On ppc, we only have MMIO no specific IO space from the CPU

     * perspective.  In theory we ought to be able to embed the PCI IO

     * memory region direction in the system memory space.  However,

     * if any of the IO BAR subregions use the old_portio mechanism,

     * that won't be processed properly unless accessed from the

     * system io address space.  This hack to bounce things via

     * system_io works around the problem until all the users of

     * old_portion are updated */

    sprintf(namebuf, "%s.io", sphb->dtbusname);

    memory_region_init(&sphb->iospace, namebuf, SPAPR_PCI_IO_WIN_SIZE);

    /* FIXME: fix to support multiple PHBs */

    memory_region_add_subregion(get_system_io(), 0, &sphb->iospace);



    sprintf(namebuf, "%s.io-alias", sphb->dtbusname);

    memory_region_init_io(&sphb->iowindow, &spapr_io_ops, sphb,

                          namebuf, SPAPR_PCI_IO_WIN_SIZE);

    memory_region_add_subregion(get_system_memory(), sphb->io_win_addr,

                                &sphb->iowindow);



    /* As MSI/MSIX interrupts trigger by writing at MSI/MSIX vectors,

     * we need to allocate some memory to catch those writes coming

     * from msi_notify()/msix_notify() */

    if (msi_supported) {

        sprintf(namebuf, "%s.msi", sphb->dtbusname);

        memory_region_init_io(&sphb->msiwindow, &spapr_msi_ops, sphb,

                              namebuf, SPAPR_MSIX_MAX_DEVS * 0x10000);

        memory_region_add_subregion(get_system_memory(), sphb->msi_win_addr,

                                    &sphb->msiwindow);

    }



    bus = pci_register_bus(DEVICE(s),

                           sphb->busname ? sphb->busname : sphb->dtbusname,

                           pci_spapr_set_irq, pci_spapr_map_irq, sphb,

                           &sphb->memspace, &sphb->iospace,

                           PCI_DEVFN(0, 0), PCI_NUM_PINS);

    phb->bus = bus;



    sphb->dma_liobn = SPAPR_PCI_BASE_LIOBN | (pci_find_domain(bus) << 16);

    sphb->dma_window_start = 0;

    sphb->dma_window_size = 0x40000000;

    sphb->dma = spapr_tce_new_dma_context(sphb->dma_liobn, sphb->dma_window_size);

    pci_setup_iommu(bus, spapr_pci_dma_context_fn, sphb);



    QLIST_INSERT_HEAD(&spapr->phbs, sphb, list);



    /* Initialize the LSI table */

    for (i = 0; i < PCI_NUM_PINS; i++) {

        uint32_t irq;



        irq = spapr_allocate_lsi(0);

        if (!irq) {

            return -1;

        }



        sphb->lsi_table[i].irq = irq;

    }



    return 0;

}
