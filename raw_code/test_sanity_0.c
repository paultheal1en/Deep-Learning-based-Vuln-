static void test_sanity(void)

{

    AHCIQState *ahci;

    ahci = ahci_boot();

    ahci_shutdown(ahci);

}
