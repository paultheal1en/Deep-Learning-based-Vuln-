static void vfio_err_notifier_handler(void *opaque)

{

    VFIOPCIDevice *vdev = opaque;



    if (!event_notifier_test_and_clear(&vdev->err_notifier)) {

        return;

    }



    /*

     * TBD. Retrieve the error details and decide what action

     * needs to be taken. One of the actions could be to pass

     * the error to the guest and have the guest driver recover

     * from the error. This requires that PCIe capabilities be

     * exposed to the guest. For now, we just terminate the

     * guest to contain the error.

     */



    error_report("%s(%04x:%02x:%02x.%x) Unrecoverable error detected.  "

                 "Please collect any data possible and then kill the guest",

                 __func__, vdev->host.domain, vdev->host.bus,

                 vdev->host.slot, vdev->host.function);



    vm_stop(RUN_STATE_INTERNAL_ERROR);

}
