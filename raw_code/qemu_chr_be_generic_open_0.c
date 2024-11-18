void qemu_chr_be_generic_open(CharDriverState *s)

{

    if (s->idle_tag == 0) {

        s->idle_tag = g_idle_add(qemu_chr_be_generic_open_bh, s);

    }

}
