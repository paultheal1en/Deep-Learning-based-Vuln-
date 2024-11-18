static int load_uboot_image(const char *filename, hwaddr *ep, hwaddr *loadaddr,

                            int *is_linux, uint8_t image_type,

                            uint64_t (*translate_fn)(void *, uint64_t),

                            void *translate_opaque)

{

    int fd;

    int size;

    hwaddr address;

    uboot_image_header_t h;

    uboot_image_header_t *hdr = &h;

    uint8_t *data = NULL;

    int ret = -1;

    int do_uncompress = 0;



    fd = open(filename, O_RDONLY | O_BINARY);

    if (fd < 0)

        return -1;



    size = read(fd, hdr, sizeof(uboot_image_header_t));

    if (size < 0)

        goto out;



    bswap_uboot_header(hdr);



    if (hdr->ih_magic != IH_MAGIC)

        goto out;



    if (hdr->ih_type != image_type) {

        fprintf(stderr, "Wrong image type %d, expected %d\n", hdr->ih_type,

                image_type);

        goto out;

    }



    /* TODO: Implement other image types.  */

    switch (hdr->ih_type) {

    case IH_TYPE_KERNEL:

        address = hdr->ih_load;

        if (translate_fn) {

            address = translate_fn(translate_opaque, address);

        }

        if (loadaddr) {

            *loadaddr = hdr->ih_load;

        }



        switch (hdr->ih_comp) {

        case IH_COMP_NONE:

            break;

        case IH_COMP_GZIP:

            do_uncompress = 1;

            break;

        default:

            fprintf(stderr,

                    "Unable to load u-boot images with compression type %d\n",

                    hdr->ih_comp);

            goto out;

        }



        if (ep) {

            *ep = hdr->ih_ep;

        }



        /* TODO: Check CPU type.  */

        if (is_linux) {

            if (hdr->ih_os == IH_OS_LINUX) {

                *is_linux = 1;

            } else {

                *is_linux = 0;

            }

        }



        break;

    case IH_TYPE_RAMDISK:

        address = *loadaddr;

        break;

    default:

        fprintf(stderr, "Unsupported u-boot image type %d\n", hdr->ih_type);

        goto out;

    }



    data = g_malloc(hdr->ih_size);



    if (read(fd, data, hdr->ih_size) != hdr->ih_size) {

        fprintf(stderr, "Error reading file\n");

        goto out;

    }



    if (do_uncompress) {

        uint8_t *compressed_data;

        size_t max_bytes;

        ssize_t bytes;



        compressed_data = data;

        max_bytes = UBOOT_MAX_GUNZIP_BYTES;

        data = g_malloc(max_bytes);



        bytes = gunzip(data, max_bytes, compressed_data, hdr->ih_size);

        g_free(compressed_data);

        if (bytes < 0) {

            fprintf(stderr, "Unable to decompress gzipped image!\n");

            goto out;

        }

        hdr->ih_size = bytes;

    }



    rom_add_blob_fixed(filename, data, hdr->ih_size, address);



    ret = hdr->ih_size;



out:

    if (data)

        g_free(data);

    close(fd);

    return ret;

}
