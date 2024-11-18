void ide_atapi_cmd(IDEState *s)

{

    uint8_t *buf = s->io_buffer;

    const struct AtapiCmd *cmd = &atapi_cmd_table[s->io_buffer[0]];



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

    if (s->sense_key == UNIT_ATTENTION && !(cmd->flags & ALLOW_UA)) {

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

    if (!(cmd->flags & ALLOW_UA) &&

        !s->tray_open && blk_is_inserted(s->blk) && s->cdrom_changed) {



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

    if ((cmd->flags & CHECK_READY) &&

        (!media_present(s) || !blk_is_inserted(s->blk)))

    {

        ide_atapi_cmd_error(s, NOT_READY, ASC_MEDIUM_NOT_PRESENT);

        return;

    }



    /* Nondata commands permit the byte_count_limit to be 0.

     * If this is a data-transferring PIO command and BCL is 0,

     * we abort at the /ATA/ level, not the ATAPI level.

     * See ATA8 ACS3 section 7.17.6.49 and 7.21.5 */

    if (cmd->handler && !(cmd->flags & NONDATA)) {

        /* TODO: Check IDENTIFY data word 125 for default BCL (currently 0) */

        if (!(atapi_byte_count_limit(s) || s->atapi_dma)) {

            /* TODO: Move abort back into core.c and make static inline again */

            ide_abort_command(s);

            return;

        }

    }



    /* Execute the command */

    if (cmd->handler) {

        cmd->handler(s, buf);

        return;

    }



    ide_atapi_cmd_error(s, ILLEGAL_REQUEST, ASC_ILLEGAL_OPCODE);

}
