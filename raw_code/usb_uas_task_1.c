static void usb_uas_task(UASDevice *uas, uas_ui *ui)

{

    uint16_t tag = be16_to_cpu(ui->hdr.tag);

    uint64_t lun64 = be64_to_cpu(ui->task.lun);

    SCSIDevice *dev = usb_uas_get_dev(uas, lun64);

    int lun = usb_uas_get_lun(lun64);

    UASRequest *req;

    uint16_t task_tag;






    req = usb_uas_find_request(uas, be16_to_cpu(ui->hdr.tag));

    if (req) {

        goto overlapped_tag;


    if (dev == NULL) {

        goto incorrect_lun;




    switch (ui->task.function) {

    case UAS_TMF_ABORT_TASK:

        task_tag = be16_to_cpu(ui->task.task_tag);

        trace_usb_uas_tmf_abort_task(uas->dev.addr, tag, task_tag);

        req = usb_uas_find_request(uas, task_tag);

        if (req && req->dev == dev) {

            scsi_req_cancel(req->req);


        usb_uas_queue_response(uas, tag, UAS_RC_TMF_COMPLETE, 0);

        break;



    case UAS_TMF_LOGICAL_UNIT_RESET:

        trace_usb_uas_tmf_logical_unit_reset(uas->dev.addr, tag, lun);

        qdev_reset_all(&dev->qdev);

        usb_uas_queue_response(uas, tag, UAS_RC_TMF_COMPLETE, 0);

        break;



    default:

        trace_usb_uas_tmf_unsupported(uas->dev.addr, tag, ui->task.function);

        usb_uas_queue_response(uas, tag, UAS_RC_TMF_NOT_SUPPORTED, 0);

        break;


    return;



invalid_tag:

    usb_uas_queue_response(uas, tag, UAS_RC_INVALID_INFO_UNIT, 0);

    return;



overlapped_tag:

    usb_uas_queue_response(uas, req->tag, UAS_RC_OVERLAPPED_TAG, 0);

    return;



incorrect_lun:

    usb_uas_queue_response(uas, tag, UAS_RC_INCORRECT_LUN, 0);
