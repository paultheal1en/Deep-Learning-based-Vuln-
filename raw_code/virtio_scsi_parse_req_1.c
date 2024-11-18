static int virtio_scsi_parse_req(VirtIOSCSIReq *req,

                                 unsigned req_size, unsigned resp_size)

{

    VirtIODevice *vdev = (VirtIODevice *) req->dev;

    size_t in_size, out_size;



    if (iov_to_buf(req->elem.out_sg, req->elem.out_num, 0,

                   &req->req, req_size) < req_size) {

        return -EINVAL;

    }



    if (qemu_iovec_concat_iov(&req->resp_iov,

                              req->elem.in_sg, req->elem.in_num, 0,

                              resp_size) < resp_size) {

        return -EINVAL;

    }



    req->resp_size = resp_size;



    /* Old BIOSes left some padding by mistake after the req_size/resp_size.

     * As a workaround, always consider the first buffer as the virtio-scsi

     * request/response, making the payload start at the second element

     * of the iovec.

     *

     * The actual length of the response header, stored in req->resp_size,

     * does not change.

     *

     * TODO: always disable this workaround for virtio 1.0 devices.

     */

    if (!virtio_has_feature(vdev, VIRTIO_F_ANY_LAYOUT)) {

        req_size = req->elem.out_sg[0].iov_len;

        resp_size = req->elem.in_sg[0].iov_len;

    }



    out_size = qemu_sgl_concat(req, req->elem.out_sg,

                               &req->elem.out_addr[0], req->elem.out_num,

                               req_size);

    in_size = qemu_sgl_concat(req, req->elem.in_sg,

                              &req->elem.in_addr[0], req->elem.in_num,

                              resp_size);



    if (out_size && in_size) {

        return -ENOTSUP;

    }



    if (out_size) {

        req->mode = SCSI_XFER_TO_DEV;

    } else if (in_size) {

        req->mode = SCSI_XFER_FROM_DEV;

    }



    return 0;

}
