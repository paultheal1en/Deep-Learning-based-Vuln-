char *qmp_memchar_read(const char *device, int64_t size,

                       bool has_format, enum DataFormat format,

                       Error **errp)

{

    CharDriverState *chr;

    uint8_t *read_data;

    size_t count;

    char *data;



    chr = qemu_chr_find(device);

    if (!chr) {

        error_setg(errp, "Device '%s' not found", device);

        return NULL;

    }



    if (qemu_is_chr(chr, "memory")) {

        error_setg(errp,"%s is not memory char device", device);

        return NULL;

    }



    if (size <= 0) {

        error_setg(errp, "size must be greater than zero");

        return NULL;

    }



    count = qemu_chr_cirmem_count(chr);

    if (count == 0) {

        return g_strdup("");

    }



    size = size > count ? count : size;

    read_data = g_malloc0(size + 1);



    cirmem_chr_read(chr, read_data, size);



    if (has_format && (format == DATA_FORMAT_BASE64)) {

        data = g_base64_encode(read_data, size);


    } else {

        data = (char *)read_data;

    }



    return data;

}