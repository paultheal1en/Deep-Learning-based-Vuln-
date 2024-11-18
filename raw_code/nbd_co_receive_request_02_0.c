static ssize_t nbd_co_receive_request(NBDRequest *req,

                                      struct nbd_request *request)

{

    NBDClient *client = req->client;

    uint32_t command;

    ssize_t rc;



    g_assert(qemu_in_coroutine());

    client->recv_coroutine = qemu_coroutine_self();

    nbd_update_can_read(client);



    rc = nbd_receive_request(client->ioc, request);

    if (rc < 0) {

        if (rc != -EAGAIN) {

            rc = -EIO;

        }

        goto out;

    }



    TRACE("Decoding type");



    command = request->type & NBD_CMD_MASK_COMMAND;

    if (command != NBD_CMD_WRITE) {

        /* No payload, we are ready to read the next request.  */

        req->complete = true;

    }



    if (command == NBD_CMD_DISC) {

        /* Special case: we're going to disconnect without a reply,

         * whether or not flags, from, or len are bogus */

        TRACE("Request type is DISCONNECT");

        rc = -EIO;

        goto out;

    }



    /* Check for sanity in the parameters, part 1.  Defer as many

     * checks as possible until after reading any NBD_CMD_WRITE

     * payload, so we can try and keep the connection alive.  */

    if ((request->from + request->len) < request->from) {

        LOG("integer overflow detected, you're probably being attacked");

        rc = -EINVAL;

        goto out;

    }



    if (command == NBD_CMD_READ || command == NBD_CMD_WRITE) {

        if (request->len > NBD_MAX_BUFFER_SIZE) {

            LOG("len (%" PRIu32" ) is larger than max len (%u)",

                request->len, NBD_MAX_BUFFER_SIZE);

            rc = -EINVAL;

            goto out;

        }



        req->data = blk_try_blockalign(client->exp->blk, request->len);

        if (req->data == NULL) {

            rc = -ENOMEM;

            goto out;

        }

    }

    if (command == NBD_CMD_WRITE) {

        TRACE("Reading %" PRIu32 " byte(s)", request->len);



        if (read_sync(client->ioc, req->data, request->len) != request->len) {

            LOG("reading from socket failed");

            rc = -EIO;

            goto out;

        }

        req->complete = true;

    }



    /* Sanity checks, part 2. */

    if (request->from + request->len > client->exp->size) {

        LOG("operation past EOF; From: %" PRIu64 ", Len: %" PRIu32

            ", Size: %" PRIu64, request->from, request->len,

            (uint64_t)client->exp->size);

        rc = command == NBD_CMD_WRITE ? -ENOSPC : -EINVAL;

        goto out;

    }

    if (request->type & ~NBD_CMD_MASK_COMMAND & ~NBD_CMD_FLAG_FUA) {

        LOG("unsupported flags (got 0x%x)",

            request->type & ~NBD_CMD_MASK_COMMAND);

        rc = -EINVAL;

        goto out;

    }



    rc = 0;



out:

    client->recv_coroutine = NULL;

    nbd_update_can_read(client);



    return rc;

}
