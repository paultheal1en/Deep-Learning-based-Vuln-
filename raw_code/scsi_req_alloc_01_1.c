SCSIRequest *scsi_req_alloc(const SCSIReqOps *reqops, SCSIDevice *d,

                            uint32_t tag, uint32_t lun, void *hba_private)

{

    SCSIRequest *req;

    SCSIBus *bus = scsi_bus_from_device(d);

    BusState *qbus = BUS(bus);



    req = g_malloc0(reqops->size);

    req->refcount = 1;

    req->bus = bus;

    req->dev = d;

    req->tag = tag;

    req->lun = lun;

    req->hba_private = hba_private;

    req->status = -1;

    req->sense_len = 0;

    req->ops = reqops;

    object_ref(OBJECT(d));

    object_ref(OBJECT(qbus->parent));

    trace_scsi_req_alloc(req->dev->id, req->lun, req->tag);

    return req;

}
