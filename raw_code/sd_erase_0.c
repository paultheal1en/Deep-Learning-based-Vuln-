static void sd_erase(SDState *sd)

{

    int i, start, end;

    if (!sd->erase_start || !sd->erase_end) {

        sd->card_status |= ERASE_SEQ_ERROR;

        return;

    }



    start = sd->erase_start >>

            (HWBLOCK_SHIFT + SECTOR_SHIFT + WPGROUP_SHIFT);

    end = sd->erase_end >>

            (HWBLOCK_SHIFT + SECTOR_SHIFT + WPGROUP_SHIFT);

    sd->erase_start = 0;

    sd->erase_end = 0;

    sd->csd[14] |= 0x40;



    for (i = start; i <= end; i ++)

        if (sd->wp_groups[i])

            sd->card_status |= WP_ERASE_SKIP;

}
