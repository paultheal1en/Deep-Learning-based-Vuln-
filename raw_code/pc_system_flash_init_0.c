static void pc_system_flash_init(MemoryRegion *rom_memory)

{

    int unit;

    DriveInfo *pflash_drv;

    BlockDriverState *bdrv;

    int64_t size;

    char *fatal_errmsg = NULL;

    hwaddr phys_addr = 0x100000000ULL;

    int sector_bits, sector_size;

    pflash_t *system_flash;

    MemoryRegion *flash_mem;

    char name[64];



    sector_bits = 12;

    sector_size = 1 << sector_bits;



    for (unit = 0;

         (unit < FLASH_MAP_UNIT_MAX &&

          (pflash_drv = drive_get(IF_PFLASH, 0, unit)) != NULL);

         ++unit) {

        bdrv = blk_bs(blk_by_legacy_dinfo(pflash_drv));

        size = bdrv_getlength(bdrv);

        if (size < 0) {

            fatal_errmsg = g_strdup_printf("failed to get backing file size");

        } else if (size == 0) {

            fatal_errmsg = g_strdup_printf("PC system firmware (pflash) "

                               "cannot have zero size");

        } else if ((size % sector_size) != 0) {

            fatal_errmsg = g_strdup_printf("PC system firmware (pflash) "

                               "must be a multiple of 0x%x", sector_size);

        } else if (phys_addr < size || phys_addr - size < FLASH_MAP_BASE_MIN) {

            fatal_errmsg = g_strdup_printf("oversized backing file, pflash "

                               "segments cannot be mapped under "

                               TARGET_FMT_plx, FLASH_MAP_BASE_MIN);

        }

        if (fatal_errmsg != NULL) {

            Location loc;



            /* push a new, "none" location on the location stack; overwrite its

             * contents with the location saved in the option; print the error

             * (includes location); pop the top

             */

            loc_push_none(&loc);

            if (pflash_drv->opts != NULL) {

                qemu_opts_loc_restore(pflash_drv->opts);

            }

            error_report("%s", fatal_errmsg);

            loc_pop(&loc);

            g_free(fatal_errmsg);

            exit(1);

        }



        phys_addr -= size;



        /* pflash_cfi01_register() creates a deep copy of the name */

        snprintf(name, sizeof name, "system.flash%d", unit);

        system_flash = pflash_cfi01_register(phys_addr, NULL /* qdev */, name,

                                             size, bdrv, sector_size,

                                             size >> sector_bits,

                                             1      /* width */,

                                             0x0000 /* id0 */,

                                             0x0000 /* id1 */,

                                             0x0000 /* id2 */,

                                             0x0000 /* id3 */,

                                             0      /* be */);

        if (unit == 0) {

            flash_mem = pflash_cfi01_get_memory(system_flash);

            pc_isa_bios_init(rom_memory, flash_mem, size);

        }

    }

}
