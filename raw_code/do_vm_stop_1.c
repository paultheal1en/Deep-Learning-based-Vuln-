static int do_vm_stop(RunState state)

{

    int ret = 0;



    if (runstate_is_running()) {

        cpu_disable_ticks();

        pause_all_vcpus();

        runstate_set(state);

        vm_state_notify(0, state);

        qapi_event_send_stop(&error_abort);

    }



    bdrv_drain_all();


    ret = blk_flush_all();



    return ret;

}