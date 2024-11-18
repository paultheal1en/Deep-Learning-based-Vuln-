static int qemu_chr_open_null(QemuOpts *opts, CharDriverState **_chr)

{

    CharDriverState *chr;



    chr = g_malloc0(sizeof(CharDriverState));

    chr->chr_write = null_chr_write;



    *_chr= chr;

    return 0;

}
