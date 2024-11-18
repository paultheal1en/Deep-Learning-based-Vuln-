static void usbredir_chardev_open(USBRedirDevice *dev)

{

    uint32_t caps[USB_REDIR_CAPS_SIZE] = { 0, };

    char version[32];

    int flags = 0;



    /* Make sure any pending closes are handled (no-op if none pending) */

    usbredir_chardev_close_bh(dev);

    qemu_bh_cancel(dev->chardev_close_bh);



    DPRINTF("creating usbredirparser\n");



    strcpy(version, "qemu usb-redir guest ");

    pstrcat(version, sizeof(version), qemu_get_version());



    dev->parser = qemu_oom_check(usbredirparser_create());

    dev->parser->priv = dev;

    dev->parser->log_func = usbredir_log;

    dev->parser->read_func = usbredir_read;

    dev->parser->write_func = usbredir_write;

    dev->parser->hello_func = usbredir_hello;

    dev->parser->device_connect_func = usbredir_device_connect;

    dev->parser->device_disconnect_func = usbredir_device_disconnect;

    dev->parser->interface_info_func = usbredir_interface_info;

    dev->parser->ep_info_func = usbredir_ep_info;

    dev->parser->configuration_status_func = usbredir_configuration_status;

    dev->parser->alt_setting_status_func = usbredir_alt_setting_status;

    dev->parser->iso_stream_status_func = usbredir_iso_stream_status;

    dev->parser->interrupt_receiving_status_func =

        usbredir_interrupt_receiving_status;

    dev->parser->bulk_streams_status_func = usbredir_bulk_streams_status;

    dev->parser->control_packet_func = usbredir_control_packet;

    dev->parser->bulk_packet_func = usbredir_bulk_packet;

    dev->parser->iso_packet_func = usbredir_iso_packet;

    dev->parser->interrupt_packet_func = usbredir_interrupt_packet;

    dev->read_buf = NULL;

    dev->read_buf_size = 0;



    usbredirparser_caps_set_cap(caps, usb_redir_cap_connect_device_version);

    usbredirparser_caps_set_cap(caps, usb_redir_cap_filter);

    usbredirparser_caps_set_cap(caps, usb_redir_cap_ep_info_max_packet_size);

    usbredirparser_caps_set_cap(caps, usb_redir_cap_64bits_ids);



    if (runstate_check(RUN_STATE_INMIGRATE)) {

        flags |= usbredirparser_fl_no_hello;

    }

    usbredirparser_init(dev->parser, version, caps, USB_REDIR_CAPS_SIZE,

                        flags);

    usbredirparser_do_write(dev->parser);

}
