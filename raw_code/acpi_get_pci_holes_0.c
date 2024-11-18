static void acpi_get_pci_holes(Range *hole, Range *hole64)

{

    Object *pci_host;



    pci_host = acpi_get_i386_pci_host();

    g_assert(pci_host);



    hole->begin = object_property_get_int(pci_host,

                                          PCI_HOST_PROP_PCI_HOLE_START,

                                          NULL);

    hole->end = object_property_get_int(pci_host,

                                        PCI_HOST_PROP_PCI_HOLE_END,

                                        NULL);

    hole64->begin = object_property_get_int(pci_host,

                                            PCI_HOST_PROP_PCI_HOLE64_START,

                                            NULL);

    hole64->end = object_property_get_int(pci_host,

                                          PCI_HOST_PROP_PCI_HOLE64_END,

                                          NULL);

}
