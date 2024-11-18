static void *mptsas_load_request(QEMUFile *f, SCSIRequest *sreq)

{

    SCSIBus *bus = sreq->bus;

    MPTSASState *s = container_of(bus, MPTSASState, bus);

    PCIDevice *pci = PCI_DEVICE(s);

    MPTSASRequest *req;

    int i, n;



    req = g_new(MPTSASRequest, 1);

    qemu_get_buffer(f, (unsigned char *)&req->scsi_io, sizeof(req->scsi_io));



    n = qemu_get_be32(f);

    /* TODO: add a way for SCSIBusInfo's load_request to fail,

     * and fail migration instead of asserting here.

     * When we do, we might be able to re-enable NDEBUG below.

     */

#ifdef NDEBUG

#error building with NDEBUG is not supported

#endif

    assert(n >= 0);



    pci_dma_sglist_init(&req->qsg, pci, n);

    for (i = 0; i < n; i++) {

        uint64_t base = qemu_get_be64(f);

        uint64_t len = qemu_get_be64(f);

        qemu_sglist_add(&req->qsg, base, len);

    }



    scsi_req_ref(sreq);

    req->sreq = sreq;

    req->dev = s;



    return req;

}
