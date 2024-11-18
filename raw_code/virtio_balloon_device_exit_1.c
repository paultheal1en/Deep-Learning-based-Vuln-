static int virtio_balloon_device_exit(DeviceState *qdev)

{

    VirtIOBalloon *s = VIRTIO_BALLOON(qdev);

    VirtIODevice *vdev = VIRTIO_DEVICE(qdev);



    balloon_stats_destroy_timer(s);

    qemu_remove_balloon_handler(s);

    unregister_savevm(qdev, "virtio-balloon", s);

    virtio_cleanup(vdev);

    return 0;

}
