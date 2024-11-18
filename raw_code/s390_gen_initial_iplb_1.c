static bool s390_gen_initial_iplb(S390IPLState *ipl)
{
    DeviceState *dev_st;
    dev_st = get_boot_device(0);
    if (dev_st) {
        VirtioCcwDevice *virtio_ccw_dev = (VirtioCcwDevice *)
            object_dynamic_cast(OBJECT(qdev_get_parent_bus(dev_st)->parent),
                TYPE_VIRTIO_CCW_DEVICE);
        SCSIDevice *sd = (SCSIDevice *) object_dynamic_cast(OBJECT(dev_st),
                                                            TYPE_SCSI_DEVICE);
        if (virtio_ccw_dev) {
            CcwDevice *ccw_dev = CCW_DEVICE(virtio_ccw_dev);
            ipl->iplb.len = cpu_to_be32(S390_IPLB_MIN_CCW_LEN);
            ipl->iplb.blk0_len =
                cpu_to_be32(S390_IPLB_MIN_CCW_LEN - S390_IPLB_HEADER_LEN);
            ipl->iplb.pbt = S390_IPL_TYPE_CCW;
            ipl->iplb.ccw.devno = cpu_to_be16(ccw_dev->sch->devno);
            ipl->iplb.ccw.ssid = ccw_dev->sch->ssid & 3;
            return true;
        } else if (sd) {
            SCSIBus *bus = scsi_bus_from_device(sd);
            VirtIOSCSI *vdev = container_of(bus, VirtIOSCSI, bus);
            VirtIOSCSICcw *scsi_ccw = container_of(vdev, VirtIOSCSICcw, vdev);
            CcwDevice *ccw_dev = CCW_DEVICE(scsi_ccw);
            ipl->iplb.len = cpu_to_be32(S390_IPLB_MIN_QEMU_SCSI_LEN);
            ipl->iplb.blk0_len =
                cpu_to_be32(S390_IPLB_MIN_QEMU_SCSI_LEN - S390_IPLB_HEADER_LEN);
            ipl->iplb.pbt = S390_IPL_TYPE_QEMU_SCSI;
            ipl->iplb.scsi.lun = cpu_to_be32(sd->lun);
            ipl->iplb.scsi.target = cpu_to_be16(sd->id);
            ipl->iplb.scsi.channel = cpu_to_be16(sd->channel);
            ipl->iplb.scsi.devno = cpu_to_be16(ccw_dev->sch->devno);
            ipl->iplb.scsi.ssid = ccw_dev->sch->ssid & 3;
            return true;
    return false;