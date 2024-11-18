bool runstate_needs_reset(void)

{

    return runstate_check(RUN_STATE_INTERNAL_ERROR) ||

        runstate_check(RUN_STATE_SHUTDOWN) ||

        runstate_check(RUN_STATE_GUEST_PANICKED);

}
