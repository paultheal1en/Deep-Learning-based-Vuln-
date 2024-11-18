static void test_pci_spec(void)

{

    AHCIQState *ahci;

    ahci = ahci_boot();

    ahci_test_pci_spec(ahci);

    ahci_shutdown(ahci);

}
