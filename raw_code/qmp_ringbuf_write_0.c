void qmp_ringbuf_write(const char *device, const char *data,

                       bool has_format, enum DataFormat format,

                       Error **errp)

{

    CharDriverState *chr;

    const uint8_t *write_data;

    int ret;

    gsize write_count;



    chr = qemu_chr_find(device);

    if (!chr) {

        error_setg(errp, "Device '%s' not found", device);

        return;

    }



    if (!chr_is_ringbuf(chr)) {

        error_setg(errp,"%s is not a ringbuf device", device);

        return;

    }



    if (has_format && (format == DATA_FORMAT_BASE64)) {

        write_data = g_base64_decode(data, &write_count);

    } else {

        write_data = (uint8_t *)data;

        write_count = strlen(data);

    }



    ret = ringbuf_chr_write(chr, write_data, write_count);



    if (write_data != (uint8_t *)data) {

        g_free((void *)write_data);

    }



    if (ret < 0) {

        error_setg(errp, "Failed to write to device %s", device);

        return;

    }

}
