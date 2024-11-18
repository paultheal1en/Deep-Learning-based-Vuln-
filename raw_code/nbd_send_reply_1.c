static int nbd_send_reply(int csock, struct nbd_reply *reply)

{

    uint8_t buf[4 + 4 + 8];



    /* Reply

       [ 0 ..  3]    magic   (NBD_REPLY_MAGIC)

       [ 4 ..  7]    error   (0 == no error)

       [ 7 .. 15]    handle

     */

    cpu_to_be32w((uint32_t*)buf, NBD_REPLY_MAGIC);

    cpu_to_be32w((uint32_t*)(buf + 4), reply->error);

    cpu_to_be64w((uint64_t*)(buf + 8), reply->handle);



    TRACE("Sending response to client");



    if (write_sync(csock, buf, sizeof(buf)) != sizeof(buf)) {

        LOG("writing to socket failed");

        errno = EINVAL;

        return -1;

    }

    return 0;

}
