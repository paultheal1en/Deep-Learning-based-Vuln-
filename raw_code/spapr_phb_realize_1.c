static void spapr_phb_realize(DeviceState *dev, Error **errp)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    SysBusDevice *s = SYS_BUS_DEVICE(dev);

    sPAPRPHBState *sphb = SPAPR_PCI_HOST_BRIDGE(s);

    PCIHostState *phb = PCI_HOST_BRIDGE(s);

    sPAPRPHBClass *info = SPAPR_PCI_HOST_BRIDGE_GET_CLASS(s);

    char *namebuf;

    int i;

    PCIBus *bus;

    uint64_t msi_window_size = 4096;



    if (sphb->index != (uint32_t)-1) {

        hwaddr windows_base;



        if ((sphb->buid != (uint64_t)-1) || (sphb->dma_liobn != (uint32_t)-1)

            || (sphb->mem_win_addr != (hwaddr)-1)

            || (sphb->io_win_addr != (hwaddr)-1)) {

            error_setg(errp, "Either \"index\" or other parameters must"

                       " be specified for PAPR PHB, not both");

            return;

        }



        if (sphb->index > SPAPR_PCI_MAX_INDEX) {

            error_setg(errp, "\"index\" for PAPR PHB is too large (max %u)",

                       SPAPR_PCI_MAX_INDEX);

            return;

        }



        sphb->buid = SPAPR_PCI_BASE_BUID + sphb->index;

        sphb->dma_liobn = SPAPR_PCI_LIOBN(sphb->index, 0);



        windows_base = SPAPR_PCI_WINDOW_BASE

            + sphb->index * SPAPR_PCI_WINDOW_SPACING;

        sphb->mem_win_addr = windows_base + SPAPR_PCI_MMIO_WIN_OFF;

        sphb->io_win_addr = windows_base + SPAPR_PCI_IO_WIN_OFF;

    }



    if (sphb->buid == (uint64_t)-1) {

        error_setg(errp, "BUID not specified for PHB");

        return;

    }



    if (sphb->dma_liobn == (uint32_t)-1) {

        error_setg(errp, "LIOBN not specified for PHB");

        return;

    }



    if (sphb->mem_win_addr == (hwaddr)-1) {

        error_setg(errp, "Memory window address not specified for PHB");

        return;

    }



    if (sphb->io_win_addr == (hwaddr)-1) {

        error_setg(errp, "IO window address not specified for PHB");

        return;

    }



    if (spapr_pci_find_phb(spapr, sphb->buid)) {

        error_setg(errp, "PCI host bridges must have unique BUIDs");

        return;

    }



    sphb->dtbusname = g_strdup_printf("pci@%" PRIx64, sphb->buid);



    namebuf = alloca(strlen(sphb->dtbusname) + 32);



    /* Initialize memory regions */

    sprintf(namebuf, "%s.mmio", sphb->dtbusname);

    memory_region_init(&sphb->memspace, OBJECT(sphb), namebuf, UINT64_MAX);



    sprintf(namebuf, "%s.mmio-alias", sphb->dtbusname);

    memory_region_init_alias(&sphb->memwindow, OBJECT(sphb),

                             namebuf, &sphb->memspace,

                             SPAPR_PCI_MEM_WIN_BUS_OFFSET, sphb->mem_win_size);

    memory_region_add_subregion(get_system_memory(), sphb->mem_win_addr,

                                &sphb->memwindow);



    /* Initialize IO regions */

    sprintf(namebuf, "%s.io", sphb->dtbusname);

    memory_region_init(&sphb->iospace, OBJECT(sphb),

                       namebuf, SPAPR_PCI_IO_WIN_SIZE);



    sprintf(namebuf, "%s.io-alias", sphb->dtbusname);

    memory_region_init_alias(&sphb->iowindow, OBJECT(sphb), namebuf,

                             &sphb->iospace, 0, SPAPR_PCI_IO_WIN_SIZE);

    memory_region_add_subregion(get_system_memory(), sphb->io_win_addr,

                                &sphb->iowindow);



    bus = pci_register_bus(dev, NULL,

                           pci_spapr_set_irq, pci_spapr_map_irq, sphb,

                           &sphb->memspace, &sphb->iospace,

                           PCI_DEVFN(0, 0), PCI_NUM_PINS, TYPE_PCI_BUS);

    phb->bus = bus;

    qbus_set_hotplug_handler(BUS(phb->bus), DEVICE(sphb), NULL);



    /*

     * Initialize PHB address space.

     * By default there will be at least one subregion for default

     * 32bit DMA window.

     * Later the guest might want to create another DMA window

     * which will become another memory subregion.

     */

    sprintf(namebuf, "%s.iommu-root", sphb->dtbusname);



    memory_region_init(&sphb->iommu_root, OBJECT(sphb),

                       namebuf, UINT64_MAX);

    address_space_init(&sphb->iommu_as, &sphb->iommu_root,

                       sphb->dtbusname);



    /*

     * As MSI/MSIX interrupts trigger by writing at MSI/MSIX vectors,

     * we need to allocate some memory to catch those writes coming

     * from msi_notify()/msix_notify().

     * As MSIMessage:addr is going to be the same and MSIMessage:data

     * is going to be a VIRQ number, 4 bytes of the MSI MR will only

     * be used.

     *

     * For KVM we want to ensure that this memory is a full page so that

     * our memory slot is of page size granularity.

     */

#ifdef CONFIG_KVM

    if (kvm_enabled()) {

        msi_window_size = getpagesize();

    }

#endif



    memory_region_init_io(&sphb->msiwindow, NULL, &spapr_msi_ops, spapr,

                          "msi", msi_window_size);

    memory_region_add_subregion(&sphb->iommu_root, SPAPR_PCI_MSI_WINDOW,

                                &sphb->msiwindow);



    pci_setup_iommu(bus, spapr_pci_dma_iommu, sphb);



    pci_bus_set_route_irq_fn(bus, spapr_route_intx_pin_to_irq);



    QLIST_INSERT_HEAD(&spapr->phbs, sphb, list);



    /* Initialize the LSI table */

    for (i = 0; i < PCI_NUM_PINS; i++) {

        uint32_t irq;



        irq = xics_alloc_block(spapr->icp, 0, 1, true, false);

        if (!irq) {

            error_setg(errp, "spapr_allocate_lsi failed");

            return;

        }



        sphb->lsi_table[i].irq = irq;

    }



    /* allocate connectors for child PCI devices */

    if (sphb->dr_enabled) {

        for (i = 0; i < PCI_SLOT_MAX * 8; i++) {

            spapr_dr_connector_new(OBJECT(phb),

                                   SPAPR_DR_CONNECTOR_TYPE_PCI,

                                   (sphb->index << 16) | i);

        }

    }



    if (!info->finish_realize) {

        error_setg(errp, "finish_realize not defined");

        return;

    }



    info->finish_realize(sphb, errp);



    sphb->msi = g_hash_table_new_full(g_int_hash, g_int_equal, g_free, g_free);

}
