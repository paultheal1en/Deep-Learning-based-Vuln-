static void virtser_port_device_realize(DeviceState *dev, Error **errp)

{

    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);

    VirtIOSerialPortClass *vsc = VIRTIO_SERIAL_PORT_GET_CLASS(port);

    VirtIOSerialBus *bus = VIRTIO_SERIAL_BUS(qdev_get_parent_bus(dev));

    VirtIODevice *vdev = VIRTIO_DEVICE(bus->vser);

    int max_nr_ports;

    bool plugging_port0;

    Error *err = NULL;



    port->vser = bus->vser;

    port->bh = qemu_bh_new(flush_queued_data_bh, port);



    assert(vsc->have_data);



    /*

     * Is the first console port we're seeing? If so, put it up at

     * location 0. This is done for backward compatibility (old

     * kernel, new qemu).

     */

    plugging_port0 = vsc->is_console && !find_port_by_id(port->vser, 0);



    if (find_port_by_id(port->vser, port->id)) {

        error_setg(errp, "virtio-serial-bus: A port already exists at id %u",

                   port->id);

        return;

    }



    if (find_port_by_name(port->name)) {

        error_setg(errp, "virtio-serial-bus: A port already exists by name %s",

                   port->name);

        return;

    }



    if (port->id == VIRTIO_CONSOLE_BAD_ID) {

        if (plugging_port0) {

            port->id = 0;

        } else {

            port->id = find_free_port_id(port->vser);

            if (port->id == VIRTIO_CONSOLE_BAD_ID) {

                error_setg(errp, "virtio-serial-bus: Maximum port limit for "

                                 "this device reached");

                return;

            }

        }

    }



    max_nr_ports = virtio_tswap32(vdev, port->vser->config.max_nr_ports);

    if (port->id >= max_nr_ports) {

        error_setg(errp, "virtio-serial-bus: Out-of-range port id specified, "

                         "max. allowed: %u", max_nr_ports - 1);

        return;

    }



    vsc->realize(dev, &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }



    port->elem.out_num = 0;

}
