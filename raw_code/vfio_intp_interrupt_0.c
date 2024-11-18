static void vfio_intp_interrupt(VFIOINTp *intp)

{

    int ret;

    VFIOINTp *tmp;

    VFIOPlatformDevice *vdev = intp->vdev;

    bool delay_handling = false;



    qemu_mutex_lock(&vdev->intp_mutex);

    if (intp->state == VFIO_IRQ_INACTIVE) {

        QLIST_FOREACH(tmp, &vdev->intp_list, next) {

            if (tmp->state == VFIO_IRQ_ACTIVE ||

                tmp->state == VFIO_IRQ_PENDING) {

                delay_handling = true;

                break;

            }

        }

    }

    if (delay_handling) {

        /*

         * the new IRQ gets a pending status and is pushed in

         * the pending queue

         */

        intp->state = VFIO_IRQ_PENDING;

        trace_vfio_intp_interrupt_set_pending(intp->pin);

        QSIMPLEQ_INSERT_TAIL(&vdev->pending_intp_queue,

                             intp, pqnext);

        ret = event_notifier_test_and_clear(&intp->interrupt);

        qemu_mutex_unlock(&vdev->intp_mutex);

        return;

    }



    trace_vfio_platform_intp_interrupt(intp->pin,

                              event_notifier_get_fd(&intp->interrupt));



    ret = event_notifier_test_and_clear(&intp->interrupt);

    if (!ret) {

        error_report("Error when clearing fd=%d (ret = %d)",

                     event_notifier_get_fd(&intp->interrupt), ret);

    }



    intp->state = VFIO_IRQ_ACTIVE;



    /* sets slow path */

    vfio_mmap_set_enabled(vdev, false);



    /* trigger the virtual IRQ */

    qemu_set_irq(intp->qemuirq, 1);



    /*

     * Schedule the mmap timer which will restore fastpath when no IRQ

     * is active anymore

     */

    if (vdev->mmap_timeout) {

        timer_mod(vdev->mmap_timer,

                  qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) +

                      vdev->mmap_timeout);

    }

    qemu_mutex_unlock(&vdev->intp_mutex);

}
