VirtIOS390Device *s390_virtio_bus_find_vring(VirtIOS390Bus *bus,

                                             ram_addr_t mem,

                                             int *vq_num)

{

    BusChild *kid;

    int i;



    QTAILQ_FOREACH(kid, &bus->bus.children, sibling) {

        VirtIOS390Device *dev = (VirtIOS390Device *)kid->child;



        for(i = 0; i < VIRTIO_PCI_QUEUE_MAX; i++) {

            if (!virtio_queue_get_addr(dev->vdev, i))

                break;

            if (virtio_queue_get_addr(dev->vdev, i) == mem) {

                if (vq_num) {

                    *vq_num = i;

                }

                return dev;

            }

        }

    }



    return NULL;

}
