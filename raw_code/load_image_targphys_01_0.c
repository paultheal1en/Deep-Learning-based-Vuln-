int load_image_targphys(const char *filename,

                        target_phys_addr_t addr, uint64_t max_sz)

{

    int size;



    size = get_image_size(filename);

    if (size > max_sz) {

        return -1;

    }

    if (size > 0) {

        rom_add_file_fixed(filename, addr, -1);

    }

    return size;

}
