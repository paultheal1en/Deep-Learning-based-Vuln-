static void handle_output(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOSerial *vser;

    VirtIOSerialPort *port;

    VirtIOSerialPortInfo *info;



    vser = DO_UPCAST(VirtIOSerial, vdev, vdev);

    port = find_port_by_vq(vser, vq);

    info = port ? DO_UPCAST(VirtIOSerialPortInfo, qdev, port->dev.info) : NULL;



    if (!port || !port->host_connected || !info->have_data) {

        discard_vq_data(vq, vdev);

        return;

    }



    if (!port->throttled) {

        do_flush_queued_data(port, vq, vdev);

        return;

    }

}
