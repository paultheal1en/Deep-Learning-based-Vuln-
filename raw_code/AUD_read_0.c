int AUD_read (SWVoiceIn *sw, void *buf, int size)

{

    int bytes;



    if (!sw) {

        /* XXX: Consider options */

        return size;

    }



    if (!sw->hw->enabled) {

        dolog ("Reading from disabled voice %s\n", SW_NAME (sw));

        return 0;

    }



    bytes = sw->hw->pcm_ops->read (sw, buf, size);

    return bytes;

}
