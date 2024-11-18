static void mv88w8618_audio_write(void *opaque, target_phys_addr_t offset,

                                  uint64_t value, unsigned size)

{

    mv88w8618_audio_state *s = opaque;



    switch (offset) {

    case MP_AUDIO_PLAYBACK_MODE:

        if (value & MP_AUDIO_PLAYBACK_EN &&

            !(s->playback_mode & MP_AUDIO_PLAYBACK_EN)) {

            s->status = 0;

            s->last_free = 0;

            s->play_pos = 0;

        }

        s->playback_mode = value;

        mv88w8618_audio_clock_update(s);

        break;



    case MP_AUDIO_CLOCK_DIV:

        s->clock_div = value;

        s->last_free = 0;

        s->play_pos = 0;

        mv88w8618_audio_clock_update(s);

        break;



    case MP_AUDIO_IRQ_STATUS:

        s->status &= ~value;

        break;



    case MP_AUDIO_IRQ_ENABLE:

        s->irq_enable = value;

        if (s->status & s->irq_enable) {

            qemu_irq_raise(s->irq);

        }

        break;



    case MP_AUDIO_TX_START_LO:

        s->phys_buf = (s->phys_buf & 0xFFFF0000) | (value & 0xFFFF);

        s->target_buffer = s->phys_buf;

        s->play_pos = 0;

        s->last_free = 0;

        break;



    case MP_AUDIO_TX_THRESHOLD:

        s->threshold = (value + 1) * 4;

        break;



    case MP_AUDIO_TX_START_HI:

        s->phys_buf = (s->phys_buf & 0xFFFF) | (value << 16);

        s->target_buffer = s->phys_buf;

        s->play_pos = 0;

        s->last_free = 0;

        break;

    }

}
