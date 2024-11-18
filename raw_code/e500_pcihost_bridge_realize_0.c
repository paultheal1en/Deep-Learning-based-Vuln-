static void e500_pcihost_bridge_realize(PCIDevice *d, Error **errp)

{

    PPCE500PCIBridgeState *b = PPC_E500_PCI_BRIDGE(d);

    PPCE500CCSRState *ccsr = CCSR(container_get(qdev_get_machine(),

                                  "/e500-ccsr"));



    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_PCI);

    d->config[PCI_HEADER_TYPE] =

        (d->config[PCI_HEADER_TYPE] & PCI_HEADER_TYPE_MULTI_FUNCTION) |

        PCI_HEADER_TYPE_BRIDGE;



    memory_region_init_alias(&b->bar0, OBJECT(ccsr), "e500-pci-bar0", &ccsr->ccsr_space,

                             0, int128_get64(ccsr->ccsr_space.size));

    pci_register_bar(d, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &b->bar0);

}
