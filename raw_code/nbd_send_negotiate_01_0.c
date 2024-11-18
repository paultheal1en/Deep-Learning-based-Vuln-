static int nbd_send_negotiate(int csock, off_t size, uint32_t flags)

{

    char buf[8 + 8 + 8 + 128];



    /* Negotiate

        [ 0 ..   7]   passwd   ("NBDMAGIC")

        [ 8 ..  15]   magic    (0x00420281861253)

        [16 ..  23]   size

        [24 ..  27]   flags

        [28 .. 151]   reserved (0)

     */



    TRACE("Beginning negotiation.");

    memcpy(buf, "NBDMAGIC", 8);

    cpu_to_be64w((uint64_t*)(buf + 8), 0x00420281861253LL);

    cpu_to_be64w((uint64_t*)(buf + 16), size);

    cpu_to_be32w((uint32_t*)(buf + 24),

                 flags | NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_TRIM |

                 NBD_FLAG_SEND_FLUSH | NBD_FLAG_SEND_FUA);

    memset(buf + 28, 0, 124);



    if (write_sync(csock, buf, sizeof(buf)) != sizeof(buf)) {

        LOG("write failed");

        errno = EINVAL;

        return -1;

    }



    TRACE("Negotiation succeeded.");



    return 0;

}
