void qemu_chr_initial_reset(void)

{

    CharDriverState *chr;



    initial_reset_issued = 1;



    TAILQ_FOREACH(chr, &chardevs, next) {

        qemu_chr_reset(chr);

    }

}
