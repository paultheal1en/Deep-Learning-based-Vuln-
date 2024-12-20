static int nbd_negotiate_handle_info(NBDClient *client, uint32_t length,

                                     uint32_t opt, uint16_t myflags,

                                     Error **errp)

{

    int rc;

    char name[NBD_MAX_NAME_SIZE + 1];

    NBDExport *exp;

    uint16_t requests;

    uint16_t request;

    uint32_t namelen;

    bool sendname = false;

    bool blocksize = false;

    uint32_t sizes[3];

    char buf[sizeof(uint64_t) + sizeof(uint16_t)];

    const char *msg;



    /* Client sends:

        4 bytes: L, name length (can be 0)

        L bytes: export name

        2 bytes: N, number of requests (can be 0)

        N * 2 bytes: N requests

    */

    if (length < sizeof(namelen) + sizeof(requests)) {

        msg = "overall request too short";

        goto invalid;

    }

    if (nbd_read(client->ioc, &namelen, sizeof(namelen), errp) < 0) {

        return -EIO;

    }

    be32_to_cpus(&namelen);

    length -= sizeof(namelen);

    if (namelen > length - sizeof(requests) || (length - namelen) % 2) {

        msg = "name length is incorrect";

        goto invalid;

    }

    if (nbd_read(client->ioc, name, namelen, errp) < 0) {

        return -EIO;

    }

    name[namelen] = '\0';

    length -= namelen;

    trace_nbd_negotiate_handle_export_name_request(name);



    if (nbd_read(client->ioc, &requests, sizeof(requests), errp) < 0) {

        return -EIO;

    }

    be16_to_cpus(&requests);

    length -= sizeof(requests);

    trace_nbd_negotiate_handle_info_requests(requests);

    if (requests != length / sizeof(request)) {

        msg = "incorrect number of  requests for overall length";

        goto invalid;

    }

    while (requests--) {

        if (nbd_read(client->ioc, &request, sizeof(request), errp) < 0) {

            return -EIO;

        }

        be16_to_cpus(&request);

        length -= sizeof(request);

        trace_nbd_negotiate_handle_info_request(request,

                                                nbd_info_lookup(request));

        /* We care about NBD_INFO_NAME and NBD_INFO_BLOCK_SIZE;

         * everything else is either a request we don't know or

         * something we send regardless of request */

        switch (request) {

        case NBD_INFO_NAME:

            sendname = true;

            break;

        case NBD_INFO_BLOCK_SIZE:

            blocksize = true;

            break;

        }

    }



    exp = nbd_export_find(name);

    if (!exp) {

        return nbd_negotiate_send_rep_err(client->ioc, NBD_REP_ERR_UNKNOWN,

                                          opt, errp, "export '%s' not present",

                                          name);

    }



    /* Don't bother sending NBD_INFO_NAME unless client requested it */

    if (sendname) {

        rc = nbd_negotiate_send_info(client, opt, NBD_INFO_NAME, length, name,

                                     errp);

        if (rc < 0) {

            return rc;

        }

    }



    /* Send NBD_INFO_DESCRIPTION only if available, regardless of

     * client request */

    if (exp->description) {

        size_t len = strlen(exp->description);



        rc = nbd_negotiate_send_info(client, opt, NBD_INFO_DESCRIPTION,

                                     len, exp->description, errp);

        if (rc < 0) {

            return rc;

        }

    }



    /* Send NBD_INFO_BLOCK_SIZE always, but tweak the minimum size

     * according to whether the client requested it, and according to

     * whether this is OPT_INFO or OPT_GO. */

    /* minimum - 1 for back-compat, or 512 if client is new enough.

     * TODO: consult blk_bs(blk)->bl.request_alignment? */

    sizes[0] = (opt == NBD_OPT_INFO || blocksize) ? BDRV_SECTOR_SIZE : 1;

    /* preferred - Hard-code to 4096 for now.

     * TODO: is blk_bs(blk)->bl.opt_transfer appropriate? */

    sizes[1] = 4096;

    /* maximum - At most 32M, but smaller as appropriate. */

    sizes[2] = MIN(blk_get_max_transfer(exp->blk), NBD_MAX_BUFFER_SIZE);

    trace_nbd_negotiate_handle_info_block_size(sizes[0], sizes[1], sizes[2]);

    cpu_to_be32s(&sizes[0]);

    cpu_to_be32s(&sizes[1]);

    cpu_to_be32s(&sizes[2]);

    rc = nbd_negotiate_send_info(client, opt, NBD_INFO_BLOCK_SIZE,

                                 sizeof(sizes), sizes, errp);

    if (rc < 0) {

        return rc;

    }



    /* Send NBD_INFO_EXPORT always */

    trace_nbd_negotiate_new_style_size_flags(exp->size,

                                             exp->nbdflags | myflags);

    stq_be_p(buf, exp->size);

    stw_be_p(buf + 8, exp->nbdflags | myflags);

    rc = nbd_negotiate_send_info(client, opt, NBD_INFO_EXPORT,

                                 sizeof(buf), buf, errp);

    if (rc < 0) {

        return rc;

    }



    /* If the client is just asking for NBD_OPT_INFO, but forgot to

     * request block sizes, return an error.

     * TODO: consult blk_bs(blk)->request_align, and only error if it

     * is not 1? */

    if (opt == NBD_OPT_INFO && !blocksize) {

        return nbd_negotiate_send_rep_err(client->ioc,

                                          NBD_REP_ERR_BLOCK_SIZE_REQD, opt,

                                          errp,

                                          "request NBD_INFO_BLOCK_SIZE to "

                                          "use this export");

    }



    /* Final reply */

    rc = nbd_negotiate_send_rep(client->ioc, NBD_REP_ACK, opt, errp);

    if (rc < 0) {

        return rc;

    }



    if (opt == NBD_OPT_GO) {

        client->exp = exp;

        QTAILQ_INSERT_TAIL(&client->exp->clients, client, next);

        nbd_export_get(client->exp);

        rc = 1;

    }

    return rc;



 invalid:

    if (nbd_drop(client->ioc, length, errp) < 0) {

        return -EIO;

    }

    return nbd_negotiate_send_rep_err(client->ioc, NBD_REP_ERR_INVALID, opt,

                                      errp, "%s", msg);

}
