int load_image_targphys(const char *filename,

			target_phys_addr_t addr, int max_sz)

{

    FILE *f;

    size_t got;



    f = fopen(filename, "rb");

    if (!f) return -1;



    got = fread_targphys(addr, max_sz, f);

    if (ferror(f)) { fclose(f); return -1; }

    fclose(f);



    return got;

}
