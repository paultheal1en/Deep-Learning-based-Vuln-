static void test_identify(void)

{

    AHCIQState *ahci;



    ahci = ahci_boot_and_enable();

    ahci_test_identify(ahci);

    ahci_shutdown(ahci);

}
