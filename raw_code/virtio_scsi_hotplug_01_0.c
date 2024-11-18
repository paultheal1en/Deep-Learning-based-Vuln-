static void virtio_scsi_hotplug(SCSIBus *bus, SCSIDevice *dev)

{

    VirtIOSCSI *s = container_of(bus, VirtIOSCSI, bus);



    if (((s->vdev.guest_features >> VIRTIO_SCSI_F_HOTPLUG) & 1) &&

        (s->vdev.status & VIRTIO_CONFIG_S_DRIVER_OK)) {

        virtio_scsi_push_event(s, dev, VIRTIO_SCSI_T_TRANSPORT_RESET,

                               VIRTIO_SCSI_EVT_RESET_RESCAN);

    }

}
