static int usbredir_check_filter(USBRedirDevice *dev)

{

    if (dev->interface_info.interface_count == 0) {

        ERROR("No interface info for device\n");

        goto error;

    }



    if (dev->filter_rules) {

        if (!usbredirparser_peer_has_cap(dev->parser,

                                    usb_redir_cap_connect_device_version)) {

            ERROR("Device filter specified and peer does not have the "

                  "connect_device_version capability\n");

            goto error;

        }



        if (usbredirfilter_check(

                dev->filter_rules,

                dev->filter_rules_count,

                dev->device_info.device_class,

                dev->device_info.device_subclass,

                dev->device_info.device_protocol,

                dev->interface_info.interface_class,

                dev->interface_info.interface_subclass,

                dev->interface_info.interface_protocol,

                dev->interface_info.interface_count,

                dev->device_info.vendor_id,

                dev->device_info.product_id,

                dev->device_info.device_version_bcd,

                0) != 0) {

            goto error;

        }

    }



    return 0;



error:

    usbredir_device_disconnect(dev);

    if (usbredirparser_peer_has_cap(dev->parser, usb_redir_cap_filter)) {

        usbredirparser_send_filter_reject(dev->parser);

        usbredirparser_do_write(dev->parser);

    }

    return -1;

}
