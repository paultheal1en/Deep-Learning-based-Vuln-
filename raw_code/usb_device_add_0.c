static int usb_device_add(const char *devname, int is_hotplug)

{

    const char *p;

    USBDevice *dev;



    if (!free_usb_ports)

        return -1;



    if (strstart(devname, "host:", &p)) {

        dev = usb_host_device_open(p);

    } else if (!strcmp(devname, "mouse")) {

        dev = usb_mouse_init();

    } else if (!strcmp(devname, "tablet")) {

        dev = usb_tablet_init();

    } else if (!strcmp(devname, "keyboard")) {

        dev = usb_keyboard_init();

    } else if (strstart(devname, "disk:", &p)) {

        BlockDriverState *bs;



        dev = usb_msd_init(p, &bs);

        if (!dev)

            return -1;

        if (bdrv_key_required(bs)) {

            autostart = 0;

            if (is_hotplug && monitor_read_bdrv_key(bs) < 0) {

                dev->handle_destroy(dev);

                return -1;

            }

        }

    } else if (!strcmp(devname, "wacom-tablet")) {

        dev = usb_wacom_init();

    } else if (strstart(devname, "serial:", &p)) {

        dev = usb_serial_init(p);

#ifdef CONFIG_BRLAPI

    } else if (!strcmp(devname, "braille")) {

        dev = usb_baum_init();

#endif

    } else if (strstart(devname, "net:", &p)) {

        int nic = nb_nics;



        if (net_client_init("nic", p) < 0)

            return -1;

        nd_table[nic].model = "usb";

        dev = usb_net_init(&nd_table[nic]);

    } else if (!strcmp(devname, "bt") || strstart(devname, "bt:", &p)) {

        dev = usb_bt_init(devname[2] ? hci_init(p) :

                        bt_new_hci(qemu_find_bt_vlan(0)));

    } else {

        return -1;

    }

    if (!dev)

        return -1;



    return usb_device_add_dev(dev);

}
