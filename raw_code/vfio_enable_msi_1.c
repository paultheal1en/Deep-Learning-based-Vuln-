static void vfio_enable_msi(VFIODevice *vdev)

{

    int ret, i;



    vfio_disable_interrupts(vdev);



    vdev->nr_vectors = msi_nr_vectors_allocated(&vdev->pdev);

retry:

    vdev->msi_vectors = g_malloc0(vdev->nr_vectors * sizeof(VFIOMSIVector));



    for (i = 0; i < vdev->nr_vectors; i++) {

        VFIOMSIVector *vector = &vdev->msi_vectors[i];



        vector->vdev = vdev;

        vector->use = true;



        if (event_notifier_init(&vector->interrupt, 0)) {

            error_report("vfio: Error: event_notifier_init failed");

        }



        vector->msg = msi_get_message(&vdev->pdev, i);



        /*

         * Attempt to enable route through KVM irqchip,

         * default to userspace handling if unavailable.

         */

        vector->virq = kvm_irqchip_add_msi_route(kvm_state, vector->msg);

        if (vector->virq < 0 ||

            kvm_irqchip_add_irqfd_notifier(kvm_state, &vector->interrupt,

                                           NULL, vector->virq) < 0) {

            qemu_set_fd_handler(event_notifier_get_fd(&vector->interrupt),

                                vfio_msi_interrupt, NULL, vector);

        }

    }



    ret = vfio_enable_vectors(vdev, false);

    if (ret) {

        if (ret < 0) {

            error_report("vfio: Error: Failed to setup MSI fds: %m");

        } else if (ret != vdev->nr_vectors) {

            error_report("vfio: Error: Failed to enable %d "

                         "MSI vectors, retry with %d", vdev->nr_vectors, ret);

        }



        for (i = 0; i < vdev->nr_vectors; i++) {

            VFIOMSIVector *vector = &vdev->msi_vectors[i];

            if (vector->virq >= 0) {

                kvm_irqchip_remove_irqfd_notifier(kvm_state, &vector->interrupt,

                                                  vector->virq);

                kvm_irqchip_release_virq(kvm_state, vector->virq);

                vector->virq = -1;

            } else {

                qemu_set_fd_handler(event_notifier_get_fd(&vector->interrupt),

                                    NULL, NULL, NULL);

            }

            event_notifier_cleanup(&vector->interrupt);

        }



        g_free(vdev->msi_vectors);



        if (ret > 0 && ret != vdev->nr_vectors) {

            vdev->nr_vectors = ret;

            goto retry;

        }

        vdev->nr_vectors = 0;



        return;

    }



    vdev->interrupt = VFIO_INT_MSI;



    DPRINTF("%s(%04x:%02x:%02x.%x) Enabled %d MSI vectors\n", __func__,

            vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function, vdev->nr_vectors);

}
