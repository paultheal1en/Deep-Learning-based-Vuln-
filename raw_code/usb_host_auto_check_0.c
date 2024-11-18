static void usb_host_auto_check(void *unused)

{

    struct USBHostDevice *s;

    int unconnected = 0;



    usb_host_scan(NULL, usb_host_auto_scan);



    QTAILQ_FOREACH(s, &hostdevs, next) {

        if (s->fd == -1) {

            unconnected++;

        }

    }



    if (unconnected == 0) {

        /* nothing to watch */

        if (usb_auto_timer) {

            qemu_del_timer(usb_auto_timer);

        }

        return;

    }



    if (!usb_auto_timer) {

        usb_auto_timer = qemu_new_timer(rt_clock, usb_host_auto_check, NULL);

        if (!usb_auto_timer) {

            return;

        }

    }

    qemu_mod_timer(usb_auto_timer, qemu_get_clock(rt_clock) + 2000);

}
