void pcie_cap_slot_hotplug_cb(HotplugHandler *hotplug_dev, DeviceState *dev,

                              Error **errp)

{

    uint8_t *exp_cap;

    PCIDevice *pci_dev = PCI_DEVICE(dev);



    pcie_cap_slot_hotplug_common(PCI_DEVICE(hotplug_dev), dev, &exp_cap, errp);



    /* Don't send event when device is enabled during qemu machine creation:

     * it is present on boot, no hotplug event is necessary. We do send an

     * event when the device is disabled later. */

    if (!dev->hotplugged) {

        pci_word_test_and_set_mask(exp_cap + PCI_EXP_SLTSTA,

                                   PCI_EXP_SLTSTA_PDS);

        return;

    }



    /* TODO: multifunction hot-plug.

     * Right now, only a device of function = 0 is allowed to be

     * hot plugged/unplugged.

     */

    assert(PCI_FUNC(pci_dev->devfn) == 0);



    pci_word_test_and_set_mask(exp_cap + PCI_EXP_SLTSTA,

                               PCI_EXP_SLTSTA_PDS);

    pcie_cap_slot_event(PCI_DEVICE(hotplug_dev), PCI_EXP_HP_EV_PDC);

}
