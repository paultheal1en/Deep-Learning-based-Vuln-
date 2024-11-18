static void sd_reset(SDState *sd, BlockDriverState *bdrv)

{

    uint64_t size;

    uint64_t sect;



    if (bdrv) {

        bdrv_get_geometry(bdrv, &sect);

    } else {

        sect = 0;

    }

    sect <<= 9;



    size = sect + 1;



    sect = (size >> (HWBLOCK_SHIFT + SECTOR_SHIFT + WPGROUP_SHIFT)) + 1;



    sd->state = sd_idle_state;

    sd->rca = 0x0000;

    sd_set_ocr(sd);

    sd_set_scr(sd);

    sd_set_cid(sd);

    sd_set_csd(sd, size);

    sd_set_cardstatus(sd);

    sd_set_sdstatus(sd);



    sd->bdrv = bdrv;



    if (sd->wp_groups)

        qemu_free(sd->wp_groups);

    sd->wp_switch = bdrv ? bdrv_is_read_only(bdrv) : 0;

    sd->wp_groups = (int *) qemu_mallocz(sizeof(int) * sect);

    memset(sd->function_group, 0, sizeof(int) * 6);

    sd->erase_start = 0;

    sd->erase_end = 0;

    sd->size = size;

    sd->blk_len = 0x200;

    sd->pwd_len = 0;

}
