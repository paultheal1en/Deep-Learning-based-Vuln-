static void test_hba_spec(void)

{

    AHCIQState *ahci;



    ahci = ahci_boot();

    ahci_pci_enable(ahci);

    ahci_test_hba_spec(ahci);

    ahci_shutdown(ahci);

}
