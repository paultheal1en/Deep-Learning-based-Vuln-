static int pci_piix3_xen_ide_unplug(DeviceState *dev)

{

    PCIDevice *pci_dev;

    PCIIDEState *pci_ide;

    DriveInfo *di;

    int i = 0;



    pci_dev = DO_UPCAST(PCIDevice, qdev, dev);

    pci_ide = DO_UPCAST(PCIIDEState, dev, pci_dev);



    for (; i < 3; i++) {

        di = drive_get_by_index(IF_IDE, i);

        if (di != NULL && di->bdrv != NULL && !di->bdrv->removable) {

            DeviceState *ds = bdrv_get_attached(di->bdrv);

            if (ds) {

                bdrv_detach(di->bdrv, ds);

            }

            bdrv_close(di->bdrv);

            pci_ide->bus[di->bus].ifs[di->unit].bs = NULL;

            drive_put_ref(di);

        }

    }

    qdev_reset_all(&(pci_ide->dev.qdev));

    return 0;

}
