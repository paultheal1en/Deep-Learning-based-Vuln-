void vring_disable_notification(VirtIODevice *vdev, Vring *vring)

{

    if (!(vdev->guest_features & (1 << VIRTIO_RING_F_EVENT_IDX))) {

        vring_set_used_flags(vdev, vring, VRING_USED_F_NO_NOTIFY);

    }

}
