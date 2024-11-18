static int scsi_qdev_exit(DeviceState *qdev)

{

    SCSIDevice *dev = SCSI_DEVICE(qdev);



    if (dev->vmsentry) {

        qemu_del_vm_change_state_handler(dev->vmsentry);

    }

    scsi_device_destroy(dev);

    return 0;

}
