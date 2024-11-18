static int vm_can_run(void)

{

    if (powerdown_requested)

        return 0;

    if (reset_requested)

        return 0;

    if (shutdown_requested)

        return 0;

    if (debug_requested)

        return 0;

    return 1;

}
