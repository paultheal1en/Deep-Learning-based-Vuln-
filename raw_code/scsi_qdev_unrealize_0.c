static void scsi_qdev_unrealize(DeviceState *qdev, Error **errp)

{

    SCSIDevice *dev = SCSI_DEVICE(qdev);



    if (dev->vmsentry) {

        qemu_del_vm_change_state_handler(dev->vmsentry);

    }

    scsi_device_unrealize(dev, errp);

}
