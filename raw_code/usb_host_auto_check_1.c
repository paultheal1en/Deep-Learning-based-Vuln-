static void usb_host_auto_check(void *unused)

{

    struct USBHostDevice *s;

    struct USBAutoFilter *f;

    libusb_device **devs;

    struct libusb_device_descriptor ddesc;

    int unconnected = 0;

    int i, n;



    if (usb_host_init() != 0) {

        return;

    }



    if (runstate_is_running()) {

        n = libusb_get_device_list(ctx, &devs);

        for (i = 0; i < n; i++) {

            if (libusb_get_device_descriptor(devs[i], &ddesc) != 0) {

                continue;

            }

            if (ddesc.bDeviceClass == LIBUSB_CLASS_HUB) {

                continue;

            }

            QTAILQ_FOREACH(s, &hostdevs, next) {

                f = &s->match;

                if (f->bus_num > 0 &&

                    f->bus_num != libusb_get_bus_number(devs[i])) {

                    continue;

                }

                if (f->addr > 0 &&

                    f->addr != libusb_get_device_address(devs[i])) {

                    continue;

                }

                if (f->port != NULL) {

                    char port[16] = "-";

                    usb_host_get_port(devs[i], port, sizeof(port));

                    if (strcmp(f->port, port) != 0) {

                        continue;

                    }

                }

                if (f->vendor_id > 0 &&

                    f->vendor_id != ddesc.idVendor) {

                    continue;

                }

                if (f->product_id > 0 &&

                    f->product_id != ddesc.idProduct) {

                    continue;

                }



                /* We got a match */

                s->seen++;

                if (s->errcount >= 3) {

                    continue;

                }

                if (s->dh != NULL) {

                    continue;

                }

                if (usb_host_open(s, devs[i]) < 0) {

                    s->errcount++;

                    continue;

                }

                break;

            }

        }

        libusb_free_device_list(devs, 1);



        QTAILQ_FOREACH(s, &hostdevs, next) {

            if (s->dh == NULL) {

                unconnected++;

            }

            if (s->seen == 0) {

                if (s->dh) {

                    usb_host_close(s);

                }

                s->errcount = 0;

            }

            s->seen = 0;

        }



#if 0

        if (unconnected == 0) {

            /* nothing to watch */

            if (usb_auto_timer) {

                timer_del(usb_auto_timer);

                trace_usb_host_auto_scan_disabled();

            }

            return;

        }

#endif

    }



    if (!usb_vmstate) {

        usb_vmstate = qemu_add_vm_change_state_handler(usb_host_vm_state, NULL);

    }

    if (!usb_auto_timer) {

        usb_auto_timer = timer_new_ms(QEMU_CLOCK_REALTIME, usb_host_auto_check, NULL);

        if (!usb_auto_timer) {

            return;

        }

        trace_usb_host_auto_scan_enabled();

    }

    timer_mod(usb_auto_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 2000);

}
