static PCIBridgeWindows *pci_bridge_region_init(PCIBridge *br)

{

    PCIDevice *pd = PCI_DEVICE(br);

    PCIBus *parent = pd->bus;

    PCIBridgeWindows *w = g_new(PCIBridgeWindows, 1);

    uint16_t cmd = pci_get_word(pd->config + PCI_COMMAND);



    pci_bridge_init_alias(br, &w->alias_pref_mem,

                          PCI_BASE_ADDRESS_MEM_PREFETCH,

                          "pci_bridge_pref_mem",

                          &br->address_space_mem,

                          parent->address_space_mem,

                          cmd & PCI_COMMAND_MEMORY);

    pci_bridge_init_alias(br, &w->alias_mem,

                          PCI_BASE_ADDRESS_SPACE_MEMORY,

                          "pci_bridge_mem",

                          &br->address_space_mem,

                          parent->address_space_mem,

                          cmd & PCI_COMMAND_MEMORY);

    pci_bridge_init_alias(br, &w->alias_io,

                          PCI_BASE_ADDRESS_SPACE_IO,

                          "pci_bridge_io",

                          &br->address_space_io,

                          parent->address_space_io,

                          cmd & PCI_COMMAND_IO);



    pci_bridge_init_vga_aliases(br, parent, w->alias_vga);



    return w;

}
