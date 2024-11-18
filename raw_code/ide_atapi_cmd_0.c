void ide_atapi_cmd(IDEState *s)

{

    uint8_t *buf;



    buf = s->io_buffer;

#ifdef DEBUG_IDE_ATAPI

    {

        int i;

        printf("ATAPI limit=0x%x packet:", s->lcyl | (s->hcyl << 8));

        for(i = 0; i < ATAPI_PACKET_SIZE; i++) {

            printf(" %02x", buf[i]);

        }

        printf("\n");

    }

#endif

    /*

     * If there's a UNIT_ATTENTION condition pending, only command flagged with

     * ALLOW_UA are allowed to complete. with other commands getting a CHECK

     * condition response unless a higher priority status, defined by the drive

     * here, is pending.

     */

    if (s->sense_key == UNIT_ATTENTION &&

        !(atapi_cmd_table[s->io_buffer[0]].flags & ALLOW_UA)) {

        ide_atapi_cmd_check_status(s);

        return;

    }

    /*

     * When a CD gets changed, we have to report an ejected state and

     * then a loaded state to guests so that they detect tray

     * open/close and media change events.  Guests that do not use

     * GET_EVENT_STATUS_NOTIFICATION to detect such tray open/close

     * states rely on this behavior.

     */

    if (!(atapi_cmd_table[s->io_buffer[0]].flags & ALLOW_UA) &&

        !s->tray_open && bdrv_is_inserted(s->bs) && s->cdrom_changed) {



        if (s->cdrom_changed == 1) {

            ide_atapi_cmd_error(s, NOT_READY, ASC_MEDIUM_NOT_PRESENT);

            s->cdrom_changed = 2;

        } else {

            ide_atapi_cmd_error(s, UNIT_ATTENTION, ASC_MEDIUM_MAY_HAVE_CHANGED);

            s->cdrom_changed = 0;

        }



        return;

    }



    /* Report a Not Ready condition if appropriate for the command */

    if ((atapi_cmd_table[s->io_buffer[0]].flags & CHECK_READY) &&

        (!media_present(s) || !bdrv_is_inserted(s->bs)))

    {

        ide_atapi_cmd_error(s, NOT_READY, ASC_MEDIUM_NOT_PRESENT);

        return;

    }



    /* Execute the command */

    if (atapi_cmd_table[s->io_buffer[0]].handler) {

        atapi_cmd_table[s->io_buffer[0]].handler(s, buf);

        return;

    }



    ide_atapi_cmd_error(s, ILLEGAL_REQUEST, ASC_ILLEGAL_OPCODE);

}
