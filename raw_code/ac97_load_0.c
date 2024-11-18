static int ac97_load (QEMUFile *f, void *opaque, int version_id)

{

    int ret;

    size_t i;

    uint8_t active[LAST_INDEX];

    AC97LinkState *s = opaque;



    if (version_id != 2)

        return -EINVAL;



    ret = pci_device_load (s->pci_dev, f);

    if (ret)

        return ret;



    qemu_get_be32s (f, &s->glob_cnt);

    qemu_get_be32s (f, &s->glob_sta);

    qemu_get_be32s (f, &s->cas);



    for (i = 0; i < ARRAY_SIZE (s->bm_regs); ++i) {

        AC97BusMasterRegs *r = &s->bm_regs[i];

        qemu_get_be32s (f, &r->bdbar);

        qemu_get_8s (f, &r->civ);

        qemu_get_8s (f, &r->lvi);

        qemu_get_be16s (f, &r->sr);

        qemu_get_be16s (f, &r->picb);

        qemu_get_8s (f, &r->piv);

        qemu_get_8s (f, &r->cr);

        qemu_get_be32s (f, &r->bd_valid);

        qemu_get_be32s (f, &r->bd.addr);

        qemu_get_be32s (f, &r->bd.ctl_len);

    }

    qemu_get_buffer (f, s->mixer_data, sizeof (s->mixer_data));

    qemu_get_buffer (f, active, sizeof (active));



#ifdef USE_MIXER

    record_select (s, mixer_load (s, AC97_Record_Select));

#define V_(a, b) set_volume (s, a, b, mixer_load (s, a))

    V_ (AC97_Master_Volume_Mute, AUD_MIXER_VOLUME);

    V_ (AC97_PCM_Out_Volume_Mute, AUD_MIXER_PCM);

    V_ (AC97_Line_In_Volume_Mute, AUD_MIXER_LINE_IN);

#undef V_

#endif

    reset_voices (s, active);



    s->bup_flag = 0;

    s->last_samp = 0;

    return 0;

}
