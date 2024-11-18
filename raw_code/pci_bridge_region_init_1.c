static void pci_bridge_region_init(PCIBridge *br)

{

    PCIBus *parent = br->dev.bus;

    uint16_t cmd = pci_get_word(br->dev.config + PCI_COMMAND);



    pci_bridge_init_alias(br, &br->alias_pref_mem,

                          PCI_BASE_ADDRESS_MEM_PREFETCH,

                          "pci_bridge_pref_mem",

                          &br->address_space_mem,

                          parent->address_space_mem,

                          cmd & PCI_COMMAND_MEMORY);

    pci_bridge_init_alias(br, &br->alias_mem,

                          PCI_BASE_ADDRESS_SPACE_MEMORY,

                          "pci_bridge_mem",

                          &br->address_space_mem,

                          parent->address_space_mem,

                          cmd & PCI_COMMAND_MEMORY);

    pci_bridge_init_alias(br, &br->alias_io,

                          PCI_BASE_ADDRESS_SPACE_IO,

                          "pci_bridge_io",

                          &br->address_space_io,

                          parent->address_space_io,

                          cmd & PCI_COMMAND_IO);

   /* TODO: optinal VGA and VGA palette snooping support. */

}
