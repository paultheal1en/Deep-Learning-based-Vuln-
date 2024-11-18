static void vfio_intp_inject_pending_lockheld(VFIOINTp *intp)

{

    trace_vfio_platform_intp_inject_pending_lockheld(intp->pin,

                              event_notifier_get_fd(&intp->interrupt));



    intp->state = VFIO_IRQ_ACTIVE;



    /* trigger the virtual IRQ */

    qemu_set_irq(intp->qemuirq, 1);

}
