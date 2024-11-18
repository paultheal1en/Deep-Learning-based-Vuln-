VirtIODevice *virtio_serial_init(DeviceState *dev, uint32_t max_nr_ports)

{

    VirtIOSerial *vser;

    VirtIODevice *vdev;

    uint32_t i;



    if (!max_nr_ports)

        return NULL;



    vdev = virtio_common_init("virtio-serial", VIRTIO_ID_CONSOLE,

                              sizeof(struct virtio_console_config),

                              sizeof(VirtIOSerial));



    vser = DO_UPCAST(VirtIOSerial, vdev, vdev);



    /* Spawn a new virtio-serial bus on which the ports will ride as devices */

    vser->bus = virtser_bus_new(dev);

    vser->bus->vser = vser;

    QTAILQ_INIT(&vser->ports);



    vser->bus->max_nr_ports = max_nr_ports;

    vser->ivqs = qemu_malloc(max_nr_ports * sizeof(VirtQueue *));

    vser->ovqs = qemu_malloc(max_nr_ports * sizeof(VirtQueue *));



    /* Add a queue for host to guest transfers for port 0 (backward compat) */

    vser->ivqs[0] = virtio_add_queue(vdev, 128, handle_input);

    /* Add a queue for guest to host transfers for port 0 (backward compat) */

    vser->ovqs[0] = virtio_add_queue(vdev, 128, handle_output);



    /* control queue: host to guest */

    vser->c_ivq = virtio_add_queue(vdev, 16, control_in);

    /* control queue: guest to host */

    vser->c_ovq = virtio_add_queue(vdev, 16, control_out);



    for (i = 1; i < vser->bus->max_nr_ports; i++) {

        /* Add a per-port queue for host to guest transfers */

        vser->ivqs[i] = virtio_add_queue(vdev, 128, handle_input);

        /* Add a per-per queue for guest to host transfers */

        vser->ovqs[i] = virtio_add_queue(vdev, 128, handle_output);

    }



    vser->config.max_nr_ports = max_nr_ports;

    vser->ports_map = qemu_mallocz(((max_nr_ports + 31) / 32)

        * sizeof(vser->ports_map[0]));

    /*

     * Reserve location 0 for a console port for backward compat

     * (old kernel, new qemu)

     */

    mark_port_added(vser, 0);



    vser->vdev.get_features = get_features;

    vser->vdev.get_config = get_config;

    vser->vdev.set_config = set_config;



    /*

     * Register for the savevm section with the virtio-console name

     * to preserve backward compat

     */

    register_savevm(dev, "virtio-console", -1, 2, virtio_serial_save,

                    virtio_serial_load, vser);



    return vdev;

}
