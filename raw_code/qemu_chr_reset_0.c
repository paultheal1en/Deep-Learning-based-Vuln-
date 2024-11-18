void qemu_chr_reset(CharDriverState *s)

{

    if (s->bh == NULL && initial_reset_issued) {

	s->bh = qemu_bh_new(qemu_chr_reset_bh, s);

	qemu_bh_schedule(s->bh);

    }

}
