static int usb_host_scan_dev(void *opaque, USBScanFunc *func)

{

    FILE *f = NULL;

    char line[1024];

    char buf[1024];

    int bus_num, addr, speed, device_count, class_id, product_id, vendor_id;

    char product_name[512];

    int ret = 0;



    if (!usb_host_device_path) {

        perror("husb: USB Host Device Path not set");

        goto the_end;

    }

    snprintf(line, sizeof(line), "%s/devices", usb_host_device_path);

    f = fopen(line, "r");

    if (!f) {

        perror("husb: cannot open devices file");

        goto the_end;

    }



    device_count = 0;

    bus_num = addr = class_id = product_id = vendor_id = 0;

    speed = -1; /* Can't get the speed from /[proc|dev]/bus/usb/devices */

    for(;;) {

        if (fgets(line, sizeof(line), f) == NULL) {

            break;

        }

        if (strlen(line) > 0) {

            line[strlen(line) - 1] = '\0';

        }

        if (line[0] == 'T' && line[1] == ':') {

            if (device_count && (vendor_id || product_id)) {

                /* New device.  Add the previously discovered device.  */

                ret = func(opaque, bus_num, addr, 0, class_id, vendor_id,

                           product_id, product_name, speed);

                if (ret) {

                    goto the_end;

                }

            }

            if (get_tag_value(buf, sizeof(buf), line, "Bus=", " ") < 0) {

                goto fail;

            }

            bus_num = atoi(buf);

            if (get_tag_value(buf, sizeof(buf), line, "Dev#=", " ") < 0) {

                goto fail;

            }

            addr = atoi(buf);

            if (get_tag_value(buf, sizeof(buf), line, "Spd=", " ") < 0) {

                goto fail;

            }

            if (!strcmp(buf, "5000")) {

                speed = USB_SPEED_SUPER;

            } else if (!strcmp(buf, "480")) {

                speed = USB_SPEED_HIGH;

            } else if (!strcmp(buf, "1.5")) {

                speed = USB_SPEED_LOW;

            } else {

                speed = USB_SPEED_FULL;

            }

            product_name[0] = '\0';

            class_id = 0xff;

            device_count++;

            product_id = 0;

            vendor_id = 0;

        } else if (line[0] == 'P' && line[1] == ':') {

            if (get_tag_value(buf, sizeof(buf), line, "Vendor=", " ") < 0) {

                goto fail;

            }

            vendor_id = strtoul(buf, NULL, 16);

            if (get_tag_value(buf, sizeof(buf), line, "ProdID=", " ") < 0) {

                goto fail;

            }

            product_id = strtoul(buf, NULL, 16);

        } else if (line[0] == 'S' && line[1] == ':') {

            if (get_tag_value(buf, sizeof(buf), line, "Product=", "") < 0) {

                goto fail;

            }

            pstrcpy(product_name, sizeof(product_name), buf);

        } else if (line[0] == 'D' && line[1] == ':') {

            if (get_tag_value(buf, sizeof(buf), line, "Cls=", " (") < 0) {

                goto fail;

            }

            class_id = strtoul(buf, NULL, 16);

        }

    fail: ;

    }

    if (device_count && (vendor_id || product_id)) {

        /* Add the last device.  */

        ret = func(opaque, bus_num, addr, 0, class_id, vendor_id,

                   product_id, product_name, speed);

    }

 the_end:

    if (f) {

        fclose(f);

    }

    return ret;

}
