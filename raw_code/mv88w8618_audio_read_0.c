static uint64_t mv88w8618_audio_read(void *opaque, target_phys_addr_t offset,

                                    unsigned size)

{

    mv88w8618_audio_state *s = opaque;



    switch (offset) {

    case MP_AUDIO_PLAYBACK_MODE:

        return s->playback_mode;



    case MP_AUDIO_CLOCK_DIV:

        return s->clock_div;



    case MP_AUDIO_IRQ_STATUS:

        return s->status;



    case MP_AUDIO_IRQ_ENABLE:

        return s->irq_enable;



    case MP_AUDIO_TX_STATUS:

        return s->play_pos >> 2;



    default:

        return 0;

    }

}
