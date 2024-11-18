static uint8_t usb_linux_get_alt_setting(USBHostDevice *s,

    uint8_t configuration, uint8_t interface)

{

    char device_name[64], line[1024];

    int alt_setting;



    sprintf(device_name, "%d-%s:%d.%d", s->bus_num, s->port,

            (int)configuration, (int)interface);



    if (!usb_host_read_file(line, sizeof(line), "bAlternateSetting",

                            device_name)) {

        /* Assume alt 0 on error */

        return 0;

    }

    if (sscanf(line, "%d", &alt_setting) != 1) {

        /* Assume alt 0 on error */

        return 0;

    }

    return alt_setting;

}
