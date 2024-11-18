SDState *sd_init(BlockBackend *blk, bool is_spi)

{

    SDState *sd;



    if (blk && blk_is_read_only(blk)) {

        fprintf(stderr, "sd_init: Cannot use read-only drive\n");

        return NULL;

    }



    sd = (SDState *) g_malloc0(sizeof(SDState));

    sd->buf = blk_blockalign(blk, 512);

    sd->spi = is_spi;

    sd->enable = true;

    sd_reset(sd, blk);

    if (sd->blk) {

        blk_attach_dev_nofail(sd->blk, sd);

        blk_set_dev_ops(sd->blk, &sd_block_ops, sd);

    }

    vmstate_register(NULL, -1, &sd_vmstate, sd);

    return sd;

}
