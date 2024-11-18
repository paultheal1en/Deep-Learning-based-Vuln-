size_t virtio_serial_guest_ready(VirtIOSerialPort *port)

{

    VirtQueue *vq = port->ivq;

    unsigned int bytes;



    if (!virtio_queue_ready(vq) ||

        !(port->vser->vdev.status & VIRTIO_CONFIG_S_DRIVER_OK) ||

        virtio_queue_empty(vq)) {

        return 0;

    }

    if (use_multiport(port->vser) && !port->guest_connected) {

        return 0;

    }

    virtqueue_get_avail_bytes(vq, &bytes, NULL);

    return bytes;

}
