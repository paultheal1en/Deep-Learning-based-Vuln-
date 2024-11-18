static bool cmd_read_pio(IDEState *s, uint8_t cmd)

{

    bool lba48 = (cmd == WIN_READ_EXT);



    if (s->drive_kind == IDE_CD) {

        ide_set_signature(s); /* odd, but ATA4 8.27.5.2 requires it */

        ide_abort_command(s);

        return true;

    }



    if (!s->bs) {

        ide_abort_command(s);

        return true;

    }



    ide_cmd_lba48_transform(s, lba48);

    s->req_nb_sectors = 1;

    ide_sector_read(s);



    return false;

}
