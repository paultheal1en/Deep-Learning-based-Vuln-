static void pci_hotplug(void)

{

    QVirtioPCIDevice *dev;

    QOSState *qs;

    const char *arch = qtest_get_arch();



    qs = pci_test_start();



    /* plug secondary disk */

    qpci_plug_device_test("virtio-blk-pci", "drv1", PCI_SLOT_HP,

                          "'drive': 'drive1'");



    dev = virtio_blk_pci_init(qs->pcibus, PCI_SLOT_HP);

    g_assert(dev);

    qvirtio_pci_device_disable(dev);

    g_free(dev);



    /* unplug secondary disk */

    if (strcmp(arch, "i386") == 0 || strcmp(arch, "x86_64") == 0) {

        qpci_unplug_acpi_device_test("drv1", PCI_SLOT_HP);

    }

    qtest_shutdown(qs);

}
