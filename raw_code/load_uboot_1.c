int load_uboot(const char *filename, target_ulong *ep, int *is_linux)

{



    int fd;

    int size;

    uboot_image_header_t h;

    uboot_image_header_t *hdr = &h;

    uint8_t *data = NULL;



    fd = open(filename, O_RDONLY | O_BINARY);

    if (fd < 0)

        return -1;



    size = read(fd, hdr, sizeof(uboot_image_header_t));

    if (size < 0)

        goto fail;



    bswap_uboot_header(hdr);



    if (hdr->ih_magic != IH_MAGIC)

        goto fail;



    /* TODO: Implement Multi-File images.  */

    if (hdr->ih_type == IH_TYPE_MULTI) {

        fprintf(stderr, "Unable to load multi-file u-boot images\n");

        goto fail;

    }



    /* TODO: Implement compressed images.  */

    if (hdr->ih_comp != IH_COMP_NONE) {

        fprintf(stderr, "Unable to load compressed u-boot images\n");

        goto fail;

    }



    /* TODO: Check CPU type.  */

    if (is_linux) {

        if (hdr->ih_type == IH_TYPE_KERNEL && hdr->ih_os == IH_OS_LINUX)

            *is_linux = 1;

        else

            *is_linux = 0;

    }



    *ep = hdr->ih_ep;

    data = qemu_malloc(hdr->ih_size);

    if (!data)

        goto fail;



    if (read(fd, data, hdr->ih_size) != hdr->ih_size) {

        fprintf(stderr, "Error reading file\n");

        goto fail;

    }



    cpu_physical_memory_write_rom(hdr->ih_load, data, hdr->ih_size);



    return hdr->ih_size;



fail:

    if (data)

        qemu_free(data);

    close(fd);

    return -1;

}
