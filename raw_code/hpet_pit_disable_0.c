void hpet_pit_disable(void) {

    PITChannelState *s;

    s = &pit_state.channels[0];

    qemu_del_timer(s->irq_timer);

}
