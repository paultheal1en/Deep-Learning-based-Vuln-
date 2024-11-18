static int nbd_negotiate_send_rep_len(QIOChannel *ioc, uint32_t type,

                                      uint32_t opt, uint32_t len, Error **errp)

{

    uint64_t magic;



    trace_nbd_negotiate_send_rep_len(opt, type, len);



    magic = cpu_to_be64(NBD_REP_MAGIC);

    if (nbd_write(ioc, &magic, sizeof(magic), errp) < 0) {

        error_prepend(errp, "write failed (rep magic): ");

        return -EINVAL;

    }



    opt = cpu_to_be32(opt);

    if (nbd_write(ioc, &opt, sizeof(opt), errp) < 0) {

        error_prepend(errp, "write failed (rep opt): ");

        return -EINVAL;

    }



    type = cpu_to_be32(type);

    if (nbd_write(ioc, &type, sizeof(type), errp) < 0) {

        error_prepend(errp, "write failed (rep type): ");

        return -EINVAL;

    }



    len = cpu_to_be32(len);

    if (nbd_write(ioc, &len, sizeof(len), errp) < 0) {

        error_prepend(errp, "write failed (rep data length): ");

        return -EINVAL;

    }

    return 0;

}
