static int virtconsole_initfn(VirtIOSerialPort *port)

{

    VirtConsole *vcon = DO_UPCAST(VirtConsole, port, port);

    VirtIOSerialPortInfo *info = DO_UPCAST(VirtIOSerialPortInfo, qdev,

                                           vcon->port.dev.info);



    if (port->id == 0 && !info->is_console) {

        error_report("Port number 0 on virtio-serial devices reserved for virtconsole devices for backward compatibility.");

        return -1;

    }



    if (vcon->chr) {

        qemu_chr_add_handlers(vcon->chr, chr_can_read, chr_read, chr_event,

                              vcon);

        info->have_data = flush_buf;

        info->guest_open = guest_open;

        info->guest_close = guest_close;

    }



    return 0;

}
