static void do_change_block(const char *device, const char *filename)

{

    BlockDriverState *bs;



    bs = bdrv_find(device);

    if (!bs) {

        term_printf("device not found\n");

        return;

    }

    if (eject_device(bs, 0) < 0)

        return;

    bdrv_open(bs, filename, 0);

    qemu_key_check(bs, filename);

}
