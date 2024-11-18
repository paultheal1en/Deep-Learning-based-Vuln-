static AHCIQState *ahci_boot_and_enable(void)

{

    AHCIQState *ahci;

    ahci = ahci_boot();



    ahci_pci_enable(ahci);

    ahci_hba_enable(ahci);



    return ahci;

}
