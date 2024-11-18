static void usbredir_realize(USBDevice *udev, Error **errp)

{

    USBRedirDevice *dev = USB_REDIRECT(udev);

    int i;



    if (!qemu_chr_fe_get_driver(&dev->cs)) {

        error_setg(errp, QERR_MISSING_PARAMETER, "chardev");

        return;

    }



    if (dev->filter_str) {

        i = usbredirfilter_string_to_rules(dev->filter_str, ":", "|",

                                           &dev->filter_rules,

                                           &dev->filter_rules_count);

        if (i) {

            error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "filter",

                       "a usb device filter string");

            return;

        }

    }



    dev->chardev_close_bh = qemu_bh_new(usbredir_chardev_close_bh, dev);

    dev->device_reject_bh = qemu_bh_new(usbredir_device_reject_bh, dev);

    dev->attach_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, usbredir_do_attach, dev);



    packet_id_queue_init(&dev->cancelled, dev, "cancelled");

    packet_id_queue_init(&dev->already_in_flight, dev, "already-in-flight");

    usbredir_init_endpoints(dev);



    /* We'll do the attach once we receive the speed from the usb-host */

    udev->auto_attach = 0;



    /* Will be cleared during setup when we find conflicts */

    dev->compatible_speedmask = USB_SPEED_MASK_FULL | USB_SPEED_MASK_HIGH;



    /* Let the backend know we are ready */

    qemu_chr_fe_set_handlers(&dev->cs, usbredir_chardev_can_read,

                             usbredir_chardev_read, usbredir_chardev_event,

                             dev, NULL, true);



    qemu_add_vm_change_state_handler(usbredir_vm_state_change, dev);

}
