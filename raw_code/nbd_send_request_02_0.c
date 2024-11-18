ssize_t nbd_send_request(QIOChannel *ioc, struct nbd_request *request)

{

    uint8_t buf[NBD_REQUEST_SIZE];

    ssize_t ret;



    TRACE("Sending request to server: "

          "{ .from = %" PRIu64", .len = %" PRIu32 ", .handle = %" PRIu64

          ", .type=%" PRIu32 " }",

          request->from, request->len, request->handle, request->type);



    stl_be_p(buf, NBD_REQUEST_MAGIC);

    stl_be_p(buf + 4, request->type);

    stq_be_p(buf + 8, request->handle);

    stq_be_p(buf + 16, request->from);

    stl_be_p(buf + 24, request->len);



    ret = write_sync(ioc, buf, sizeof(buf));

    if (ret < 0) {

        return ret;

    }



    if (ret != sizeof(buf)) {

        LOG("writing to socket failed");

        return -EINVAL;

    }

    return 0;

}
