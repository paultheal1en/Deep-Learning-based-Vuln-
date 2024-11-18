static bool cmd_data_set_management(IDEState *s, uint8_t cmd)

{

    switch (s->feature) {

    case DSM_TRIM:

        if (s->bs) {

            ide_sector_start_dma(s, IDE_DMA_TRIM);

            return false;

        }

        break;

    }



    ide_abort_command(s);

    return true;

}
