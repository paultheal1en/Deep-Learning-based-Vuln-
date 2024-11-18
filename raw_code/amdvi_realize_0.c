static void amdvi_realize(DeviceState *dev, Error **err)

{

    int ret = 0;

    AMDVIState *s = AMD_IOMMU_DEVICE(dev);

    X86IOMMUState *x86_iommu = X86_IOMMU_DEVICE(dev);

    MachineState *ms = MACHINE(qdev_get_machine());

    MachineClass *mc = MACHINE_GET_CLASS(ms);

    PCMachineState *pcms =

        PC_MACHINE(object_dynamic_cast(OBJECT(ms), TYPE_PC_MACHINE));

    PCIBus *bus;



    if (!pcms) {

        error_setg(err, "Machine-type '%s' not supported by amd-iommu",

                   mc->name);

        return;

    }



    bus = pcms->bus;

    s->iotlb = g_hash_table_new_full(amdvi_uint64_hash,

                                     amdvi_uint64_equal, g_free, g_free);



    /* This device should take care of IOMMU PCI properties */

    x86_iommu->type = TYPE_AMD;

    qdev_set_parent_bus(DEVICE(&s->pci), &bus->qbus);

    object_property_set_bool(OBJECT(&s->pci), true, "realized", err);

    ret = pci_add_capability(&s->pci.dev, AMDVI_CAPAB_ID_SEC, 0,

                                         AMDVI_CAPAB_SIZE, err);

    if (ret < 0) {

        return;

    }

    s->capab_offset = ret;



    ret = pci_add_capability(&s->pci.dev, PCI_CAP_ID_MSI, 0,

                             AMDVI_CAPAB_REG_SIZE, err);

    if (ret < 0) {

        return;

    }

    ret = pci_add_capability(&s->pci.dev, PCI_CAP_ID_HT, 0,

                             AMDVI_CAPAB_REG_SIZE, err);

    if (ret < 0) {

        return;

    }



    /* set up MMIO */

    memory_region_init_io(&s->mmio, OBJECT(s), &mmio_mem_ops, s, "amdvi-mmio",

                          AMDVI_MMIO_SIZE);



    sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->mmio);

    sysbus_mmio_map(SYS_BUS_DEVICE(s), 0, AMDVI_BASE_ADDR);

    pci_setup_iommu(bus, amdvi_host_dma_iommu, s);

    s->devid = object_property_get_int(OBJECT(&s->pci), "addr", err);

    msi_init(&s->pci.dev, 0, 1, true, false, err);

    amdvi_init(s);

}
