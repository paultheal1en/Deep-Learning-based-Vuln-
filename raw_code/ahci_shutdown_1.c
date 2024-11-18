static void ahci_shutdown(AHCIQState *ahci)
{
    QOSState *qs = ahci->parent;
    free_ahci_device(ahci->dev);
    g_free(ahci);
    qtest_shutdown(qs);
}