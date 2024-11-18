FDCtrl *fdctrl_init_isa(DriveInfo **fds)

{

    ISADevice *dev;



    dev = isa_create("isa-fdc");

    if (fds[0]) {

        qdev_prop_set_drive_nofail(&dev->qdev, "driveA", fds[0]->bdrv);

    }

    if (fds[1]) {

        qdev_prop_set_drive_nofail(&dev->qdev, "driveB", fds[1]->bdrv);

    }

    if (qdev_init(&dev->qdev) < 0)

        return NULL;

    return &(DO_UPCAST(FDCtrlISABus, busdev, dev)->state);

}
