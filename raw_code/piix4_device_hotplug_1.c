static int piix4_device_hotplug(DeviceState *qdev, PCIDevice *dev,

				PCIHotplugState state)

{

    int slot = PCI_SLOT(dev->devfn);

    PIIX4PMState *s = DO_UPCAST(PIIX4PMState, dev,

                                PCI_DEVICE(qdev));



    /* Don't send event when device is enabled during qemu machine creation:

     * it is present on boot, no hotplug event is necessary. We do send an

     * event when the device is disabled later. */

    if (state == PCI_COLDPLUG_ENABLED) {

        return 0;

    }



    s->pci0_status.up = 0;

    s->pci0_status.down = 0;

    if (state == PCI_HOTPLUG_ENABLED) {

        enable_device(s, slot);

    } else {

        disable_device(s, slot);

    }



    pm_update_sci(s);



    return 0;

}
