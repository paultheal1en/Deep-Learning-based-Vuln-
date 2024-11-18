static void kvm_pit_put(PITCommonState *pit)

{

    KVMPITState *s = KVM_PIT(pit);

    struct kvm_pit_state2 kpit;

    struct kvm_pit_channel_state *kchan;

    struct PITChannelState *sc;

    int i, ret;



    /* The offset keeps changing as long as the VM is stopped. */

    if (s->vm_stopped) {

        kvm_pit_update_clock_offset(s);

    }



    kpit.flags = pit->channels[0].irq_disabled ? KVM_PIT_FLAGS_HPET_LEGACY : 0;

    for (i = 0; i < 3; i++) {

        kchan = &kpit.channels[i];

        sc = &pit->channels[i];

        kchan->count = sc->count;

        kchan->latched_count = sc->latched_count;

        kchan->count_latched = sc->count_latched;

        kchan->status_latched = sc->status_latched;

        kchan->status = sc->status;

        kchan->read_state = sc->read_state;

        kchan->write_state = sc->write_state;

        kchan->write_latch = sc->write_latch;

        kchan->rw_mode = sc->rw_mode;

        kchan->mode = sc->mode;

        kchan->bcd = sc->bcd;

        kchan->gate = sc->gate;

        kchan->count_load_time = sc->count_load_time - s->kernel_clock_offset;

    }



    ret = kvm_vm_ioctl(kvm_state,

                       kvm_has_pit_state2() ? KVM_SET_PIT2 : KVM_SET_PIT,

                       &kpit);

    if (ret < 0) {

        fprintf(stderr, "%s failed: %s\n",

                kvm_has_pit_state2() ? "KVM_SET_PIT2" : "KVM_SET_PIT",

                strerror(ret));

        abort();

    }

}
