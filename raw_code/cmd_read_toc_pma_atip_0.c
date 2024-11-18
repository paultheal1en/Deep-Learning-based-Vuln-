static void cmd_read_toc_pma_atip(IDEState *s, uint8_t* buf)

{

    int format, msf, start_track, len;

    uint64_t total_sectors = s->nb_sectors >> 2;

    int max_len;



    if (total_sectors == 0) {

        ide_atapi_cmd_error(s, SENSE_NOT_READY, ASC_MEDIUM_NOT_PRESENT);

        return;

    }



    max_len = ube16_to_cpu(buf + 7);

    format = buf[9] >> 6;

    msf = (buf[1] >> 1) & 1;

    start_track = buf[6];



    switch(format) {

    case 0:

        len = cdrom_read_toc(total_sectors, buf, msf, start_track);

        if (len < 0)

            goto error_cmd;

        ide_atapi_cmd_reply(s, len, max_len);

        break;

    case 1:

        /* multi session : only a single session defined */

        memset(buf, 0, 12);

        buf[1] = 0x0a;

        buf[2] = 0x01;

        buf[3] = 0x01;

        ide_atapi_cmd_reply(s, 12, max_len);

        break;

    case 2:

        len = cdrom_read_toc_raw(total_sectors, buf, msf, start_track);

        if (len < 0)

            goto error_cmd;

        ide_atapi_cmd_reply(s, len, max_len);

        break;

    default:

    error_cmd:

        ide_atapi_cmd_error(s, SENSE_ILLEGAL_REQUEST,

                            ASC_INV_FIELD_IN_CMD_PACKET);

    }

}
