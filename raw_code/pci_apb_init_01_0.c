PCIBus *pci_apb_init(hwaddr special_base,

                     hwaddr mem_base,

                     qemu_irq *ivec_irqs, PCIBus **busA, PCIBus **busB,

                     qemu_irq **pbm_irqs)

{

    DeviceState *dev;

    SysBusDevice *s;

    PCIHostState *phb;

    APBState *d;

    IOMMUState *is;

    PCIDevice *pci_dev;

    PCIBridge *br;



    /* Ultrasparc PBM main bus */

    dev = qdev_create(NULL, TYPE_APB);

    d = APB_DEVICE(dev);

    phb = PCI_HOST_BRIDGE(dev);

    phb->bus = pci_register_bus(DEVICE(phb), "pci",

                                pci_apb_set_irq, pci_pbm_map_irq, d,

                                &d->pci_mmio,

                                get_system_io(),

                                0, 32, TYPE_PCI_BUS);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    /* apb_config */

    sysbus_mmio_map(s, 0, special_base);

    /* PCI configuration space */

    sysbus_mmio_map(s, 1, special_base + 0x1000000ULL);

    /* pci_ioport */

    sysbus_mmio_map(s, 2, special_base + 0x2000000ULL);



    memory_region_init(&d->pci_mmio, OBJECT(s), "pci-mmio", 0x100000000ULL);

    memory_region_add_subregion(get_system_memory(), mem_base, &d->pci_mmio);



    *pbm_irqs = d->pbm_irqs;

    d->ivec_irqs = ivec_irqs;



    pci_create_simple(phb->bus, 0, "pbm-pci");



    /* APB IOMMU */

    is = &d->iommu;

    memset(is, 0, sizeof(IOMMUState));



    memory_region_init_iommu(&is->iommu, sizeof(is->iommu),

                             TYPE_APB_IOMMU_MEMORY_REGION, OBJECT(dev),

                             "iommu-apb", UINT64_MAX);

    address_space_init(&is->iommu_as, MEMORY_REGION(&is->iommu), "pbm-as");

    pci_setup_iommu(phb->bus, pbm_pci_dma_iommu, is);



    /* APB secondary busses */

    pci_dev = pci_create_multifunction(phb->bus, PCI_DEVFN(1, 0), true,

                                   TYPE_PBM_PCI_BRIDGE);

    br = PCI_BRIDGE(pci_dev);

    pci_bridge_map_irq(br, "pciB", pci_apb_map_irq);

    qdev_init_nofail(&pci_dev->qdev);

    *busB = pci_bridge_get_sec_bus(br);



    pci_dev = pci_create_multifunction(phb->bus, PCI_DEVFN(1, 1), true,

                                   TYPE_PBM_PCI_BRIDGE);

    br = PCI_BRIDGE(pci_dev);

    pci_bridge_map_irq(br, "pciA", pci_apb_map_irq);

    qdev_prop_set_bit(DEVICE(pci_dev), "busA", true);

    qdev_init_nofail(&pci_dev->qdev);

    *busA = pci_bridge_get_sec_bus(br);



    return phb->bus;

}
