void pstrcpy_targphys(const char *name, target_phys_addr_t dest, int buf_size,

                      const char *source)

{

    const char *nulp;

    char *ptr;



    if (buf_size <= 0) return;

    nulp = memchr(source, 0, buf_size);

    if (nulp) {

        rom_add_blob_fixed(name, source, (nulp - source) + 1, dest);

    } else {

        rom_add_blob_fixed(name, source, buf_size, dest);

        ptr = rom_ptr(dest + buf_size - 1);

        *ptr = 0;

    }

}
