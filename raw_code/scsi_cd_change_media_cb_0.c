static void scsi_cd_change_media_cb(void *opaque, bool load)

{

    SCSIDiskState *s = opaque;



    /*

     * When a CD gets changed, we have to report an ejected state and

     * then a loaded state to guests so that they detect tray

     * open/close and media change events.  Guests that do not use

     * GET_EVENT_STATUS_NOTIFICATION to detect such tray open/close

     * states rely on this behavior.

     *

     * media_changed governs the state machine used for unit attention

     * report.  media_event is used by GET EVENT STATUS NOTIFICATION.

     */

    s->media_changed = load;

    s->tray_open = !load;

    s->qdev.unit_attention = SENSE_CODE(UNIT_ATTENTION_NO_MEDIUM);

    s->media_event = true;

    s->eject_request = false;

}
