int AUD_write (SWVoiceOut *sw, void *buf, int size)

{

    int bytes;



    if (!sw) {

        /* XXX: Consider options */

        return size;

    }



    if (!sw->hw->enabled) {

        dolog ("Writing to disabled voice %s\n", SW_NAME (sw));

        return 0;

    }



    bytes = sw->hw->pcm_ops->write (sw, buf, size);

    return bytes;

}
