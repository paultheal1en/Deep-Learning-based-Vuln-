static void complete_request_vring(VirtIOBlockReq *req, unsigned char status)

{

    stb_p(&req->in->status, status);



    vring_push(&req->dev->dataplane->vring, req->elem,

               req->qiov.size + sizeof(*req->in));

    notify_guest(req->dev->dataplane);

}
