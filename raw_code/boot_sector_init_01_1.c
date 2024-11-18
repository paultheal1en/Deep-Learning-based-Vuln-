int boot_sector_init(const char *fname)

{

    FILE *f = fopen(fname, "w");

    size_t len = sizeof boot_sector;



    if (!f) {

        fprintf(stderr, "Couldn't open \"%s\": %s", fname, strerror(errno));

        return 1;

    }



    /* For Open Firmware based system, we can use a Forth script instead */

    if (strcmp(qtest_get_arch(), "ppc64") == 0) {

        len = sprintf((char *)boot_sector, "\\ Bootscript\n%x %x c! %x %x c!\n",

                LOW(SIGNATURE), BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET,

                HIGH(SIGNATURE), BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET + 1);

    }



    fwrite(boot_sector, 1, len, f);

    fclose(f);

    return 0;

}
