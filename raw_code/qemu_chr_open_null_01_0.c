static CharDriverState *qemu_chr_open_null(void)

{

    CharDriverState *chr;



    chr = g_malloc0(sizeof(CharDriverState));

    chr->chr_write = null_chr_write;

    chr->explicit_be_open = true;

    return chr;

}
