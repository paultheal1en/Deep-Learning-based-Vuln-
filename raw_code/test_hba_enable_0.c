static void test_hba_enable(void)

{

    AHCIQState *ahci;



    ahci = ahci_boot();

    ahci_pci_enable(ahci);

    ahci_hba_enable(ahci);

    ahci_shutdown(ahci);

}
