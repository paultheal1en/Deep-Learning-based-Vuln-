static void ac97_save (QEMUFile *f, void *opaque)

{

    size_t i;

    uint8_t active[LAST_INDEX];

    AC97LinkState *s = opaque;



    pci_device_save (s->pci_dev, f);



    qemu_put_be32s (f, &s->glob_cnt);

    qemu_put_be32s (f, &s->glob_sta);

    qemu_put_be32s (f, &s->cas);



    for (i = 0; i < ARRAY_SIZE (s->bm_regs); ++i) {

        AC97BusMasterRegs *r = &s->bm_regs[i];

        qemu_put_be32s (f, &r->bdbar);

        qemu_put_8s (f, &r->civ);

        qemu_put_8s (f, &r->lvi);

        qemu_put_be16s (f, &r->sr);

        qemu_put_be16s (f, &r->picb);

        qemu_put_8s (f, &r->piv);

        qemu_put_8s (f, &r->cr);

        qemu_put_be32s (f, &r->bd_valid);

        qemu_put_be32s (f, &r->bd.addr);

        qemu_put_be32s (f, &r->bd.ctl_len);

    }

    qemu_put_buffer (f, s->mixer_data, sizeof (s->mixer_data));



    active[PI_INDEX] = AUD_is_active_in (s->voice_pi) ? 1 : 0;

    active[PO_INDEX] = AUD_is_active_out (s->voice_po) ? 1 : 0;

    active[MC_INDEX] = AUD_is_active_in (s->voice_mc) ? 1 : 0;

    qemu_put_buffer (f, active, sizeof (active));

}
