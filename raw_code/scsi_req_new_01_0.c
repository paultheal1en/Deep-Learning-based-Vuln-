SCSIRequest *scsi_req_new(SCSIDevice *d, uint32_t tag, uint32_t lun,

                          void *hba_private)

{

    return d->info->alloc_req(d, tag, lun, hba_private);

}
