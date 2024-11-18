static void spapr_phb_remove_pci_device_cb(DeviceState *dev, void *opaque)

{

    /* some version guests do not wait for completion of a device

     * cleanup (generally done asynchronously by the kernel) before

     * signaling to QEMU that the device is safe, but instead sleep

     * for some 'safe' period of time. unfortunately on a busy host

     * this sleep isn't guaranteed to be long enough, resulting in

     * bad things like IRQ lines being left asserted during final

     * device removal. to deal with this we call reset just prior

     * to finalizing the device, which will put the device back into

     * an 'idle' state, as the device cleanup code expects.

     */

    pci_device_reset(PCI_DEVICE(dev));

    object_unparent(OBJECT(dev));

}
