static int qemu_shutdown_requested(void)

{

    return atomic_xchg(&shutdown_requested, 0);

}
