void acpi_pcihp_device_unplug_cb(HotplugHandler *hotplug_dev, AcpiPciHpState *s,

                                 DeviceState *dev, Error **errp)

{

    PCIDevice *pdev = PCI_DEVICE(dev);

    int slot = PCI_SLOT(pdev->devfn);

    int bsel = acpi_pcihp_get_bsel(pdev->bus);

    if (bsel < 0) {

        error_setg(errp, "Unsupported bus. Bus doesn't have property '"

                   ACPI_PCIHP_PROP_BSEL "' set");

        return;

    }



    s->acpi_pcihp_pci_status[bsel].down |= (1U << slot);

    acpi_send_event(DEVICE(hotplug_dev), ACPI_PCI_HOTPLUG_STATUS);

}
