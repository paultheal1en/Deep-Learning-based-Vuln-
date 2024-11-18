static int spapr_vga_init(PCIBus *pci_bus)

{

    switch (vga_interface_type) {

    case VGA_STD:

        pci_std_vga_init(pci_bus);

        return 1;

    case VGA_NONE:

        return 0;

    default:

        fprintf(stderr, "This vga model is not supported,"

                "currently it only supports -vga std\n");

        exit(0);

        break;

    }

}
