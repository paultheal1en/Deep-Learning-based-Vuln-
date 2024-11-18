static void usb_uas_command(UASDevice *uas, uas_ui *ui)
{
    UASRequest *req;
    uint32_t len;
    uint16_t tag = be16_to_cpu(ui->hdr.tag);
    if (uas_using_streams(uas) && tag > UAS_MAX_STREAMS) {
        goto invalid_tag;
    }
    req = usb_uas_find_request(uas, tag);
    if (req) {
        goto overlapped_tag;
    }
    req = usb_uas_alloc_request(uas, ui);
    if (req->dev == NULL) {
        goto bad_target;
    }
    trace_usb_uas_command(uas->dev.addr, req->tag,
                          usb_uas_get_lun(req->lun),
                          req->lun >> 32, req->lun & 0xffffffff);
    QTAILQ_INSERT_TAIL(&uas->requests, req, next);
    if (uas_using_streams(uas) && uas->data3[req->tag] != NULL) {
        req->data = uas->data3[req->tag];
        req->data_async = true;
        uas->data3[req->tag] = NULL;
    }
    req->req = scsi_req_new(req->dev, req->tag,
                            usb_uas_get_lun(req->lun),
                            ui->command.cdb, req);
    if (uas->requestlog) {
        scsi_req_print(req->req);
    }
    len = scsi_req_enqueue(req->req);
    if (len) {
        req->data_size = len;
        scsi_req_continue(req->req);
    }
overlapped_tag:
    usb_uas_queue_fake_sense(uas, tag, sense_code_OVERLAPPED_COMMANDS);
bad_target:
    usb_uas_queue_fake_sense(uas, tag, sense_code_LUN_NOT_SUPPORTED);
    g_free(req);
}