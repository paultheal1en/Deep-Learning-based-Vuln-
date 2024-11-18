static void virtio_scsi_handle_ctrl(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOSCSI *s = (VirtIOSCSI *)vdev;

    VirtIOSCSIReq *req;



    while ((req = virtio_scsi_pop_req(s, vq))) {

        int type;



        if (iov_to_buf(req->elem.out_sg, req->elem.out_num, 0,

                       &type, sizeof(type)) < sizeof(type)) {

            virtio_scsi_bad_req();

            continue;

        }



        tswap32s(&req->req.tmf->type);

        if (req->req.tmf->type == VIRTIO_SCSI_T_TMF) {

            if (virtio_scsi_parse_req(req, sizeof(VirtIOSCSICtrlTMFReq),

                                      sizeof(VirtIOSCSICtrlTMFResp)) < 0) {

                virtio_scsi_bad_req();

            } else {

                virtio_scsi_do_tmf(s, req);

            }



        } else if (req->req.tmf->type == VIRTIO_SCSI_T_AN_QUERY ||

                   req->req.tmf->type == VIRTIO_SCSI_T_AN_SUBSCRIBE) {

            if (virtio_scsi_parse_req(req, sizeof(VirtIOSCSICtrlANReq),

                                      sizeof(VirtIOSCSICtrlANResp)) < 0) {

                virtio_scsi_bad_req();

            } else {

                req->resp.an->event_actual = 0;

                req->resp.an->response = VIRTIO_SCSI_S_OK;

            }

        }

        virtio_scsi_complete_req(req);

    }

}
