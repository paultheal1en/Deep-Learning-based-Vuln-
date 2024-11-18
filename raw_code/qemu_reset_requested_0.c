static int qemu_reset_requested(void)

{

    int r = reset_requested;

    reset_requested = 0;

    return r;

}
