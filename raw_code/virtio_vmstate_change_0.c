static void virtio_vmstate_change(void *opaque, int running, RunState state)

{

    VirtIODevice *vdev = opaque;

    BusState *qbus = qdev_get_parent_bus(DEVICE(vdev));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    bool backend_run = running && (vdev->status & VIRTIO_CONFIG_S_DRIVER_OK);



    if (running) {

        vdev->vm_running = running;

    }



    if (backend_run) {

        virtio_set_status(vdev, vdev->status);

    }



    if (k->vmstate_change) {

        k->vmstate_change(qbus->parent, backend_run);

    }



    if (!backend_run) {

        virtio_set_status(vdev, vdev->status);

    }



    if (!running) {

        vdev->vm_running = running;

    }

}
