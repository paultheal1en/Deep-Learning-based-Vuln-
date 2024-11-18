static char *spapr_get_fw_dev_path(FWPathProvider *p, BusState *bus,

                                   DeviceState *dev)

{

#define CAST(type, obj, name) \

    ((type *)object_dynamic_cast(OBJECT(obj), (name)))

    SCSIDevice *d = CAST(SCSIDevice,  dev, TYPE_SCSI_DEVICE);

    sPAPRPHBState *phb = CAST(sPAPRPHBState, dev, TYPE_SPAPR_PCI_HOST_BRIDGE);

    VHostSCSICommon *vsc = CAST(VHostSCSICommon, dev, TYPE_VHOST_SCSI_COMMON);



    if (d) {

        void *spapr = CAST(void, bus->parent, "spapr-vscsi");

        VirtIOSCSI *virtio = CAST(VirtIOSCSI, bus->parent, TYPE_VIRTIO_SCSI);

        USBDevice *usb = CAST(USBDevice, bus->parent, TYPE_USB_DEVICE);



        if (spapr) {

            /*

             * Replace "channel@0/disk@0,0" with "disk@8000000000000000":

             * We use SRP luns of the form 8000 | (bus << 8) | (id << 5) | lun

             * in the top 16 bits of the 64-bit LUN

             */

            unsigned id = 0x8000 | (d->id << 8) | d->lun;

            return g_strdup_printf("%s@%"PRIX64, qdev_fw_name(dev),

                                   (uint64_t)id << 48);

        } else if (virtio) {

            /*

             * We use SRP luns of the form 01000000 | (target << 8) | lun

             * in the top 32 bits of the 64-bit LUN

             * Note: the quote above is from SLOF and it is wrong,

             * the actual binding is:

             * swap 0100 or 10 << or 20 << ( target lun-id -- srplun )

             */

            unsigned id = 0x1000000 | (d->id << 16) | d->lun;





            return g_strdup_printf("%s@%"PRIX64, qdev_fw_name(dev),

                                   (uint64_t)id << 32);

        } else if (usb) {

            /*

             * We use SRP luns of the form 01000000 | (usb-port << 16) | lun

             * in the top 32 bits of the 64-bit LUN

             */

            unsigned usb_port = atoi(usb->port->path);

            unsigned id = 0x1000000 | (usb_port << 16) | d->lun;

            return g_strdup_printf("%s@%"PRIX64, qdev_fw_name(dev),

                                   (uint64_t)id << 32);





    /*

     * SLOF probes the USB devices, and if it recognizes that the device is a

     * storage device, it changes its name to "storage" instead of "usb-host",

     * and additionally adds a child node for the SCSI LUN, so the correct

     * boot path in SLOF is something like .../storage@1/disk@xxx" instead.

     */

    if (strcmp("usb-host", qdev_fw_name(dev)) == 0) {

        USBDevice *usbdev = CAST(USBDevice, dev, TYPE_USB_DEVICE);

        if (usb_host_dev_is_scsi_storage(usbdev)) {

            return g_strdup_printf("storage@%s/disk", usbdev->port->path);





    if (phb) {

        /* Replace "pci" with "pci@800000020000000" */

        return g_strdup_printf("pci@%"PRIX64, phb->buid);




    if (vsc) {

        /* Same logic as virtio above */

        unsigned id = 0x1000000 | (vsc->target << 16) | vsc->lun;

        return g_strdup_printf("disk@%"PRIX64, (uint64_t)id << 32);




    if (g_str_equal("pci-bridge", qdev_fw_name(dev))) {

        /* SLOF uses "pci" instead of "pci-bridge" for PCI bridges */

        PCIDevice *pcidev = CAST(PCIDevice, dev, TYPE_PCI_DEVICE);

        return g_strdup_printf("pci@%x", PCI_SLOT(pcidev->devfn));




    return NULL;
