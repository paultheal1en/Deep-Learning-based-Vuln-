static int load_option_rom(const char *oprom, target_phys_addr_t start,

                           target_phys_addr_t end)

{

    int size;

    char *filename;



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, oprom);

    if (filename) {

        size = get_image_size(filename);

        if (size > 0 && start + size > end) {

            fprintf(stderr, "Not enough space to load option rom '%s'\n",

                    oprom);

            exit(1);

        }

        size = load_image_targphys(filename, start, end - start);

        qemu_free(filename);

    } else {

        size = -1;

    }

    if (size < 0) {

        fprintf(stderr, "Could not load option rom '%s'\n", oprom);

        exit(1);

    }

    /* Round up optiom rom size to the next 2k boundary */

    size = (size + 2047) & ~2047;

    option_rom_setup_reset(start, size);

    return size;

}
