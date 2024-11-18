static int vfio_start_eventfd_injection(VFIOINTp *intp)

{

    int ret;



    ret = vfio_set_trigger_eventfd(intp, vfio_intp_interrupt);

    if (ret) {

        error_report("vfio: Error: Failed to pass IRQ fd to the driver: %m");

    }

    return ret;

}
