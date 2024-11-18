static void flush_queued_data(VirtIOSerialPort *port, bool discard)

{

    assert(port || discard);



    do_flush_queued_data(port, port->ovq, &port->vser->vdev, discard);

}
