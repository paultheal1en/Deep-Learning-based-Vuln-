static void inet_addr_to_opts(QemuOpts *opts, const InetSocketAddress *addr)

{

    bool ipv4 = addr->ipv4 || !addr->has_ipv4;

    bool ipv6 = addr->ipv6 || !addr->has_ipv6;



    if (!ipv4 || !ipv6) {

        qemu_opt_set_bool(opts, "ipv4", ipv4, &error_abort);

        qemu_opt_set_bool(opts, "ipv6", ipv6, &error_abort);

    }

    if (addr->has_to) {

        qemu_opt_set_number(opts, "to", addr->to, &error_abort);

    }

    qemu_opt_set(opts, "host", addr->host, &error_abort);

    qemu_opt_set(opts, "port", addr->port, &error_abort);

}
