static void virtio_crypto_dataq_bh(void *opaque)

{

    VirtIOCryptoQueue *q = opaque;

    VirtIOCrypto *vcrypto = q->vcrypto;

    VirtIODevice *vdev = VIRTIO_DEVICE(vcrypto);



    /* This happens when device was stopped but BH wasn't. */

    if (!vdev->vm_running) {

        return;

    }



    /* Just in case the driver is not ready on more */

    if (unlikely(!(vdev->status & VIRTIO_CONFIG_S_DRIVER_OK))) {

        return;

    }



    virtio_crypto_handle_dataq(vdev, q->dataq);

    virtio_queue_set_notification(q->dataq, 1);

}
