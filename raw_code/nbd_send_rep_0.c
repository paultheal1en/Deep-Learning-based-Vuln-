static int nbd_send_rep(int csock, uint32_t type, uint32_t opt)

{

    uint64_t magic;

    uint32_t len;



    magic = cpu_to_be64(NBD_REP_MAGIC);

    if (write_sync(csock, &magic, sizeof(magic)) != sizeof(magic)) {

        LOG("write failed (rep magic)");

        return -EINVAL;

    }

    opt = cpu_to_be32(opt);

    if (write_sync(csock, &opt, sizeof(opt)) != sizeof(opt)) {

        LOG("write failed (rep opt)");

        return -EINVAL;

    }

    type = cpu_to_be32(type);

    if (write_sync(csock, &type, sizeof(type)) != sizeof(type)) {

        LOG("write failed (rep type)");

        return -EINVAL;

    }

    len = cpu_to_be32(0);

    if (write_sync(csock, &len, sizeof(len)) != sizeof(len)) {

        LOG("write failed (rep data length)");

        return -EINVAL;

    }

    return 0;

}
