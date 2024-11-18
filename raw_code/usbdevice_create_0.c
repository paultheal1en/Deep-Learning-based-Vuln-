USBDevice *usbdevice_create(const char *cmdline)

{

    USBBus *bus = usb_bus_find(-1 /* any */);

    DeviceInfo *info;

    USBDeviceInfo *usb;

    char driver[32];

    const char *params;

    int len;



    params = strchr(cmdline,':');

    if (params) {

        params++;

        len = params - cmdline;

        if (len > sizeof(driver))

            len = sizeof(driver);

        pstrcpy(driver, len, cmdline);

    } else {

        params = "";

        pstrcpy(driver, sizeof(driver), cmdline);

    }



    for (info = device_info_list; info != NULL; info = info->next) {

        if (info->bus_info != &usb_bus_info)

            continue;

        usb = DO_UPCAST(USBDeviceInfo, qdev, info);

        if (usb->usbdevice_name == NULL)

            continue;

        if (strcmp(usb->usbdevice_name, driver) != 0)

            continue;

        break;

    }

    if (info == NULL) {

#if 0

        /* no error because some drivers are not converted (yet) */

        error_report("usbdevice %s not found", driver);

#endif

        return NULL;

    }



    if (!usb->usbdevice_init) {

        if (params) {

            error_report("usbdevice %s accepts no params", driver);

            return NULL;

        }

        return usb_create_simple(bus, usb->qdev.name);

    }

    return usb->usbdevice_init(params);

}
