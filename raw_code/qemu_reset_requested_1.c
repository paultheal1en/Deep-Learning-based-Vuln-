static int qemu_reset_requested(void)

{

    int r = reset_requested;

    if (r && replay_checkpoint(CHECKPOINT_RESET_REQUESTED)) {

        reset_requested = 0;

        return r;

    }

    return false;

}
