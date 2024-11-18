static unsigned int event_status_media(IDEState *s,

                                       uint8_t *buf)

{

    uint8_t event_code, media_status;



    media_status = 0;

    if (s->tray_open) {

        media_status = MS_TRAY_OPEN;

    } else if (bdrv_is_inserted(s->bs)) {

        media_status = MS_MEDIA_PRESENT;

    }



    /* Event notification descriptor */

    event_code = MEC_NO_CHANGE;

    if (media_status != MS_TRAY_OPEN) {

        if (s->events.new_media) {

            event_code = MEC_NEW_MEDIA;

            s->events.new_media = false;

        } else if (s->events.eject_request) {

            event_code = MEC_EJECT_REQUESTED;

            s->events.eject_request = false;

        }

    }



    buf[4] = event_code;

    buf[5] = media_status;



    /* These fields are reserved, just clear them. */

    buf[6] = 0;

    buf[7] = 0;



    return 8; /* We wrote to 4 extra bytes from the header */

}
