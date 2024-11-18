static void pflash_timer (void *opaque)

{

    pflash_t *pfl = opaque;



    DPRINTF("%s: command %02x done\n", __func__, pfl->cmd);

    /* Reset flash */

    pfl->status ^= 0x80;

    if (pfl->bypass) {

        pfl->wcycle = 2;

    } else {

        memory_region_rom_device_set_readable(&pfl->mem, true);

        pfl->wcycle = 0;

    }

    pfl->cmd = 0;

}
