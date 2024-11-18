void qemu_chr_close(CharDriverState *chr)

{

    TAILQ_REMOVE(&chardevs, chr, next);

    if (chr->chr_close)

        chr->chr_close(chr);

    qemu_free(chr->filename);

    qemu_free(chr->label);

    qemu_free(chr);

}
