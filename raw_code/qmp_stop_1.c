void qmp_stop(Error **errp)

{

    vm_stop(RUN_STATE_PAUSED);

}
