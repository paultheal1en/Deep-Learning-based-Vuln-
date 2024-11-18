CharDriverState *qemu_chr_alloc(void)

{

    CharDriverState *chr = g_malloc0(sizeof(CharDriverState));


    return chr;

}