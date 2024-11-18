static void vfio_start_irqfd_injection(SysBusDevice *sbdev, qemu_irq irq)

{

    VFIOPlatformDevice *vdev = VFIO_PLATFORM_DEVICE(sbdev);

    VFIOINTp *intp;



    if (!kvm_irqfds_enabled() || !kvm_resamplefds_enabled() ||

        !vdev->irqfd_allowed) {

        goto fail_irqfd;

    }



    QLIST_FOREACH(intp, &vdev->intp_list, next) {

        if (intp->qemuirq == irq) {

            break;

        }

    }

    assert(intp);



    if (kvm_irqchip_add_irqfd_notifier(kvm_state, intp->interrupt,

                                   intp->unmask, irq) < 0) {

        goto fail_irqfd;

    }



    if (vfio_set_trigger_eventfd(intp, NULL) < 0) {

        goto fail_vfio;

    }

    if (vfio_set_resample_eventfd(intp) < 0) {

        goto fail_vfio;

    }



    intp->kvm_accel = true;



    trace_vfio_platform_start_irqfd_injection(intp->pin,

                                     event_notifier_get_fd(intp->interrupt),

                                     event_notifier_get_fd(intp->unmask));

    return;

fail_vfio:

    kvm_irqchip_remove_irqfd_notifier(kvm_state, intp->interrupt, irq);

    error_report("vfio: failed to start eventfd signaling for IRQ %d: %m",

                 intp->pin);

    abort();

fail_irqfd:

    vfio_start_eventfd_injection(sbdev, irq);

    return;

}
