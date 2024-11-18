static int virtio_serial_device_exit(DeviceState *dev)

{

    VirtIOSerial *vser = VIRTIO_SERIAL(dev);

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);



    unregister_savevm(dev, "virtio-console", vser);



    g_free(vser->ivqs);

    g_free(vser->ovqs);

    g_free(vser->ports_map);

    if (vser->post_load) {

        g_free(vser->post_load->connected);

        timer_del(vser->post_load->timer);

        timer_free(vser->post_load->timer);

        g_free(vser->post_load);

    }

    virtio_cleanup(vdev);

    return 0;

}
